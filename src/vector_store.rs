use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeListArray, RecordBatch, RecordBatchIterator, StringArray, UInt64Array,
    types::Float32Type,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use futures::TryStreamExt;
use lancedb::{
    Connection, DistanceType, Error as LanceError, connect,
    query::{ExecutableQuery, QueryBase},
};
use sqlx::{PgPool, Row};
use tokio::sync::RwLock;

use crate::chunk::{ChunkMatch, ChunkRecord, SearchFilter};
use crate::error::BackendError;
use crate::rag::cosine_similarity;

/// 向量存储抽象。内存实现用于默认 demo 与测试，LanceDB 实现用于持久化。
#[async_trait]
pub(crate) trait VectorStore: Send + Sync {
    async fn upsert(&self, records: Vec<ChunkRecord>) -> Result<(), BackendError>;
    async fn remove_document(&self, document_id: u64) -> Result<(), BackendError>;
    async fn search(
        &self,
        query: &[f32],
        limit: usize,
        filter: &SearchFilter,
    ) -> Result<Vec<ChunkMatch>, BackendError>;
    /// 清空所有 chunk。重置文档库时调用。
    async fn clear(&self) -> Result<(), BackendError>;
}

struct StoredChunk {
    document_id: u64,
    chunk_index: usize,
    text: String,
    vector: Vec<f32>,
}

/// 进程内向量存储，线性扫描余弦相似度。无持久化，用于离线 demo 与测试。
pub(crate) struct InMemoryVectorStore {
    chunks: RwLock<Vec<StoredChunk>>,
}

impl InMemoryVectorStore {
    pub(crate) fn new() -> Self {
        Self {
            chunks: RwLock::new(Vec::new()),
        }
    }
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn upsert(&self, records: Vec<ChunkRecord>) -> Result<(), BackendError> {
        if records.is_empty() {
            return Ok(());
        }
        let affected: HashSet<u64> = records.iter().map(|record| record.document_id).collect();
        let mut chunks = self.chunks.write().await;
        chunks.retain(|chunk| !affected.contains(&chunk.document_id));
        chunks.extend(records.into_iter().map(|record| StoredChunk {
            document_id: record.document_id,
            chunk_index: record.chunk_index,
            text: record.text,
            vector: record.vector,
        }));
        Ok(())
    }

    async fn remove_document(&self, document_id: u64) -> Result<(), BackendError> {
        self.chunks
            .write()
            .await
            .retain(|chunk| chunk.document_id != document_id);
        Ok(())
    }

    async fn clear(&self) -> Result<(), BackendError> {
        self.chunks.write().await.clear();
        Ok(())
    }

    async fn search(
        &self,
        query: &[f32],
        limit: usize,
        filter: &SearchFilter,
    ) -> Result<Vec<ChunkMatch>, BackendError> {
        let chunks = self.chunks.read().await;
        let mut matches: Vec<ChunkMatch> = chunks
            .iter()
            .filter(|chunk| filter.permits(chunk.document_id))
            .filter_map(|chunk| {
                if chunk.vector.len() != query.len() {
                    return None;
                }
                Some(ChunkMatch {
                    document_id: chunk.document_id,
                    chunk_index: chunk.chunk_index,
                    text: chunk.text.clone(),
                    score: cosine_similarity(query, &chunk.vector),
                })
            })
            .collect();
        matches.sort_by(|left, right| right.score.total_cmp(&left.score));
        matches.truncate(limit);
        Ok(matches)
    }
}

/// 基于 LanceDB 的持久化向量存储。数据落盘到本地 Lance 文件，进程重启后仍可检索。
pub(crate) struct LanceDbVectorStore {
    db: Connection,
    table_name: String,
    schema: SchemaRef,
    dim: usize,
}

impl LanceDbVectorStore {
    /// 表名按 embedding 维度区分，避免切换模型时维度不匹配的静默写入失败。
    pub(crate) async fn new(path: &str, dim: usize) -> Result<Self, BackendError> {
        let db = connect(path)
            .execute()
            .await
            .map_err(|error| BackendError::vector_store(error.to_string()))?;
        let table_name = format!("chunks_d{dim}");
        let schema = build_schema(dim);
        let store = Self {
            db,
            table_name,
            schema,
            dim,
        };
        store.ensure_table().await?;
        Ok(store)
    }

    async fn ensure_table(&self) -> Result<(), BackendError> {
        match self.db.open_table(&self.table_name).execute().await {
            Ok(_) => Ok(()),
            Err(LanceError::TableNotFound { .. }) => {
                self.db
                    .create_empty_table(&self.table_name, self.schema.clone())
                    .execute()
                    .await
                    .map_err(|error| BackendError::vector_store(error.to_string()))?;
                Ok(())
            }
            Err(error) => Err(BackendError::vector_store(error.to_string())),
        }
    }

    async fn open_table(&self) -> Result<lancedb::Table, BackendError> {
        self.db
            .open_table(&self.table_name)
            .execute()
            .await
            .map_err(|error| BackendError::vector_store(error.to_string()))
    }
}

#[async_trait]
impl VectorStore for LanceDbVectorStore {
    async fn upsert(&self, records: Vec<ChunkRecord>) -> Result<(), BackendError> {
        if records.is_empty() {
            return Ok(());
        }
        let table = self.open_table().await?;
        let affected: HashSet<u64> = records.iter().map(|record| record.document_id).collect();
        for document_id in affected {
            let predicate = format!("document_id = {document_id}");
            table
                .delete(&predicate)
                .await
                .map_err(|error| BackendError::vector_store(error.to_string()))?;
        }
        let batch = build_batch(&self.schema, &records, self.dim)?;
        let reader: Box<dyn arrow_array::RecordBatchReader + Send> = Box::new(
            RecordBatchIterator::new(vec![Ok(batch)].into_iter(), self.schema.clone()),
        );
        table
            .add(reader)
            .execute()
            .await
            .map_err(|error| BackendError::vector_store(error.to_string()))?;
        Ok(())
    }

    async fn remove_document(&self, document_id: u64) -> Result<(), BackendError> {
        let table = self.open_table().await?;
        let predicate = format!("document_id = {document_id}");
        table
            .delete(&predicate)
            .await
            .map_err(|error| BackendError::vector_store(error.to_string()))?;
        Ok(())
    }

    async fn clear(&self) -> Result<(), BackendError> {
        // 用恒真谓词删除全部行；表为空时无副作用。
        let table = self.open_table().await?;
        table
            .delete("document_id >= 0")
            .await
            .map_err(|error| BackendError::vector_store(error.to_string()))?;
        Ok(())
    }

    async fn search(
        &self,
        query: &[f32],
        limit: usize,
        filter: &SearchFilter,
    ) -> Result<Vec<ChunkMatch>, BackendError> {
        // 空范围过滤器直接短路，避免构造无意义的 IN () 谓词。
        if filter.is_empty() {
            return Ok(Vec::new());
        }
        let table = self.open_table().await?;
        let mut builder = table
            .query()
            .nearest_to(query)
            .map_err(|error| BackendError::vector_store(error.to_string()))?
            .distance_type(DistanceType::Cosine)
            .limit(limit);
        if let Some(document_ids) = &filter.document_ids {
            let ids: Vec<String> = document_ids.iter().map(|id| id.to_string()).collect();
            builder = builder.only_if(format!("document_id IN ({})", ids.join(", ")));
        }
        let stream = builder
            .execute()
            .await
            .map_err(|error| BackendError::vector_store(error.to_string()))?;
        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|error| BackendError::vector_store(error.to_string()))?;
        Ok(extract_matches(&batches))
    }
}

/// PostgreSQL + pgvector 持久化向量存储。适合数万~数十万级 chunk：
/// 向量列用 pgvector 的 `vector` 类型，检索走 `<=>`（余弦距离）并下推文档过滤。
pub(crate) struct PgVectorStore {
    pool: PgPool,
    dim: usize,
}

impl PgVectorStore {
    /// 建表（含 HNSW 索引）后返回实例。表名按维度区分（`chunks_d{dim}`），切换模型维度不串写。
    pub(crate) async fn new(database_url: &str, dim: usize) -> Result<Self, BackendError> {
        let pool = PgPool::connect(database_url)
            .await
            .map_err(|error| BackendError::vector_store(error.to_string()))?;
        let store = Self { pool, dim };
        store.ensure_schema().await?;
        Ok(store)
    }

    /// 创建 pgvector 扩展、chunks 表与 HNSW 索引（幂等）。
    async fn ensure_schema(&self) -> Result<(), BackendError> {
        let dim = self.dim;
        let table = table_name(dim);
        let sql = format!(
            "CREATE EXTENSION IF NOT EXISTS vector;
             CREATE TABLE IF NOT EXISTS {table} (
                 document_id  BIGINT NOT NULL,
                 chunk_index  INTEGER NOT NULL,
                 text         TEXT   NOT NULL,
                 vector       vector({dim}) NOT NULL,
                 PRIMARY KEY (document_id, chunk_index)
             );
             CREATE INDEX IF NOT EXISTS {table}_hnsw
             ON {table} USING hnsw (vector vector_cosine_ops);"
        );
        sqlx::raw_sql(sql.as_str())
            .execute(&self.pool)
            .await
            .map_err(|error| BackendError::vector_store(error.to_string()))?;
        Ok(())
    }
}

#[async_trait]
impl VectorStore for PgVectorStore {
    async fn upsert(&self, records: Vec<ChunkRecord>) -> Result<(), BackendError> {
        if records.is_empty() {
            return Ok(());
        }
        let table = table_name(self.dim);
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|error| BackendError::vector_store(error.to_string()))?;
        let affected: HashSet<u64> = records.iter().map(|r| r.document_id).collect();
        for document_id in &affected {
            sqlx::query(&format!("DELETE FROM {table} WHERE document_id = $1"))
                .bind(*document_id as i64)
                .execute(&mut *tx)
                .await
                .map_err(|error| BackendError::vector_store(error.to_string()))?;
        }
        let mut values_clause = String::new();
        for (i, _) in records.iter().enumerate() {
            if i > 0 {
                values_clause.push(',');
            }
            let base = i * 4;
            values_clause.push_str(&format!(
                "(${}, ${}, ${}, ${}::vector)",
                base + 1,
                base + 2,
                base + 3,
                base + 4
            ));
        }
        let insert_sql = format!(
            "INSERT INTO {table} (document_id, chunk_index, text, vector) VALUES {values_clause}"
        );
        let mut insertion = sqlx::query(&insert_sql);
        for record in &records {
            insertion = insertion
                .bind(record.document_id as i64)
                .bind(record.chunk_index as i64)
                .bind(&record.text)
                .bind(format_float_array(&record.vector));
        }
        insertion
            .execute(&mut *tx)
            .await
            .map_err(|error| BackendError::vector_store(error.to_string()))?;
        tx.commit()
            .await
            .map_err(|error| BackendError::vector_store(error.to_string()))?;
        Ok(())
    }

    async fn remove_document(&self, document_id: u64) -> Result<(), BackendError> {
        let table = table_name(self.dim);
        sqlx::query(&format!("DELETE FROM {table} WHERE document_id = $1"))
            .bind(document_id as i64)
            .execute(&self.pool)
            .await
            .map_err(|error| BackendError::vector_store(error.to_string()))?;
        Ok(())
    }

    async fn clear(&self) -> Result<(), BackendError> {
        let table = table_name(self.dim);
        sqlx::query(&format!("DELETE FROM {table}"))
            .execute(&self.pool)
            .await
            .map_err(|error| BackendError::vector_store(error.to_string()))?;
        Ok(())
    }

    async fn search(
        &self,
        query: &[f32],
        limit: usize,
        filter: &SearchFilter,
    ) -> Result<Vec<ChunkMatch>, BackendError> {
        if filter.is_empty() {
            return Ok(Vec::new());
        }
        if query.len() != self.dim {
            return Err(BackendError::vector_store(format!(
                "query dim {} != store dim {}",
                query.len(),
                self.dim
            )));
        }
        let table = table_name(self.dim);
        let filter_ids: Vec<u64> = if let Some(ids) = &filter.document_ids {
            ids.iter().copied().collect()
        } else {
            Vec::new()
        };
        // 占位符编号：$1 = 查询向量，$2..$n = 过滤文档 ID，$n+1 = limit。
        let mut sql = format!(
            "SELECT document_id, chunk_index, text,
                    vector <=> $1::vector AS dist
             FROM {table}"
        );
        let mut parameters: Vec<u64> = Vec::new();
        if !filter_ids.is_empty() {
            let placeholders = (0..filter_ids.len())
                .map(|i| format!("${}", i + 2))
                .collect::<Vec<_>>()
                .join(", ");
            sql.push_str(&format!(" WHERE document_id IN ({placeholders})"));
            parameters.extend(&filter_ids);
        }
        let limit_param = parameters.len() + 2;
        sql.push_str(&format!(
            " ORDER BY vector <=> $1::vector LIMIT ${limit_param}"
        ));
        let mut query_builder = sqlx::query(&sql).bind(format_float_array(query));
        for id in &parameters {
            query_builder = query_builder.bind(*id as i64);
        }
        query_builder = query_builder.bind(limit as i64);
        let rows = query_builder
            .fetch_all(&self.pool)
            .await
            .map_err(|error| BackendError::vector_store(error.to_string()))?;
        let mut matches = Vec::with_capacity(rows.len());
        for row in rows {
            let document_id = row.get::<i64, _>("document_id") as u64;
            let chunk_index = row.get::<i32, _>("chunk_index") as usize;
            let text = row.get::<String, _>("text");
            let dist = row.get::<f64, _>("dist") as f32;
            matches.push(ChunkMatch {
                document_id,
                chunk_index,
                text,
                score: 1.0 - dist,
            });
        }
        Ok(matches)
    }
}

fn table_name(dim: usize) -> String {
    format!("chunks_d{dim}")
}

/// 序列化 f32 向量为 pgvector 文本字面量 `[a,b,c]`。
fn format_float_array(v: &[f32]) -> String {
    let body: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    format!("[{}]", body.join(","))
}

fn build_schema(dim: usize) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("document_id", DataType::UInt64, false),
        Field::new("chunk_index", DataType::UInt64, false),
        Field::new("text", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
    ]))
}

fn build_batch(
    schema: &SchemaRef,
    records: &[ChunkRecord],
    dim: usize,
) -> Result<RecordBatch, BackendError> {
    let document_ids =
        UInt64Array::from_iter_values(records.iter().map(|record| record.document_id));
    let chunk_indices =
        UInt64Array::from_iter_values(records.iter().map(|record| record.chunk_index as u64));
    let texts = StringArray::from_iter_values(records.iter().map(|record| record.text.as_str()));
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        records
            .iter()
            .map(|record| Some(record.vector.iter().copied().map(Some).collect::<Vec<_>>())),
        dim as i32,
    );
    RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(document_ids),
            Arc::new(chunk_indices),
            Arc::new(texts),
            Arc::new(vectors),
        ],
    )
    .map_err(|error| BackendError::vector_store(error.to_string()))
}

fn extract_matches(batches: &[RecordBatch]) -> Vec<ChunkMatch> {
    let mut matches = Vec::new();
    for batch in batches {
        let Some(document_ids) = batch
            .column_by_name("document_id")
            .and_then(|column| column.as_any().downcast_ref::<UInt64Array>())
        else {
            continue;
        };
        let Some(chunk_indices) = batch
            .column_by_name("chunk_index")
            .and_then(|column| column.as_any().downcast_ref::<UInt64Array>())
        else {
            continue;
        };
        let Some(texts) = batch
            .column_by_name("text")
            .and_then(|column| column.as_any().downcast_ref::<StringArray>())
        else {
            continue;
        };
        let Some(distances) = batch
            .column_by_name("_distance")
            .and_then(|column| column.as_any().downcast_ref::<arrow_array::Float32Array>())
        else {
            continue;
        };
        for index in 0..batch.num_rows() {
            matches.push(ChunkMatch {
                document_id: document_ids.value(index),
                chunk_index: chunk_indices.value(index) as usize,
                text: texts.value(index).to_owned(),
                score: 1.0 - distances.value(index),
            });
        }
    }
    matches
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(document_id: u64, chunk_index: usize, text: &str, vector: Vec<f32>) -> ChunkRecord {
        ChunkRecord {
            document_id,
            chunk_index,
            text: text.to_owned(),
            vector,
        }
    }

    #[tokio::test]
    async fn in_memory_upsert_remove_and_search() {
        let store = InMemoryVectorStore::new();
        store
            .upsert(vec![
                record(1, 0, "rust ownership", vec![1.0, 0.0, 0.0]),
                record(1, 1, "tokio async", vec![0.0, 1.0, 0.0]),
                record(2, 0, "unrelated", vec![0.0, 0.0, 1.0]),
            ])
            .await
            .unwrap();
        let matches = store
            .search(&[1.0, 0.1, 0.0], 2, &SearchFilter::default())
            .await
            .unwrap();
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].document_id, 1);
        assert!(matches[0].text.contains("ownership"));
        store.remove_document(1).await.unwrap();
        let after = store
            .search(&[1.0, 0.0, 0.0], 10, &SearchFilter::default())
            .await
            .unwrap();
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].document_id, 2);
    }

    #[tokio::test]
    async fn in_memory_respects_document_filter() {
        let store = InMemoryVectorStore::new();
        store
            .upsert(vec![
                record(1, 0, "rust ownership", vec![1.0, 0.0, 0.0]),
                record(2, 0, "go ownership", vec![0.9, 0.1, 0.0]),
            ])
            .await
            .unwrap();
        let filter = SearchFilter {
            document_ids: Some(HashSet::from([2])),
        };
        let matches = store.search(&[1.0, 0.0, 0.0], 10, &filter).await.unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].document_id, 2);
        // 空范围过滤器短路为空结果。
        let empty = SearchFilter {
            document_ids: Some(HashSet::new()),
        };
        assert!(
            store
                .search(&[1.0, 0.0, 0.0], 10, &empty)
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn lancedb_respects_document_filter() {
        let dir =
            std::env::temp_dir().join(format!("ws-lancedb-filter-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = LanceDbVectorStore::new(dir.to_str().unwrap(), 3)
            .await
            .unwrap();
        store
            .upsert(vec![
                record(1, 0, "rust ownership", vec![1.0, 0.0, 0.0]),
                record(2, 0, "go ownership", vec![0.9, 0.1, 0.0]),
            ])
            .await
            .unwrap();
        let filter = SearchFilter {
            document_ids: Some(HashSet::from([1])),
        };
        // only_if 谓词过滤后只应返回 doc 1。
        let matches = store.search(&[1.0, 0.0, 0.0], 10, &filter).await.unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].document_id, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn lancedb_persists_and_searchs() {
        let dir = std::env::temp_dir().join(format!(
            "ws-lancedb-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let dim = 3;
        let store = LanceDbVectorStore::new(dir.to_str().unwrap(), dim)
            .await
            .unwrap();
        store
            .upsert(vec![
                record(1, 0, "rust ownership", vec![1.0, 0.0, 0.0]),
                record(2, 0, "tokio runtime", vec![0.0, 1.0, 0.0]),
            ])
            .await
            .unwrap();
        let matches = store
            .search(&[1.0, 0.05, 0.0], 1, &SearchFilter::default())
            .await
            .unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].document_id, 1);
        assert!(matches[0].score > 0.9);
        // 重新打开同一目录，验证持久化。
        drop(store);
        let reopened = LanceDbVectorStore::new(dir.to_str().unwrap(), dim)
            .await
            .unwrap();
        let again = reopened
            .search(&[0.0, 1.0, 0.0], 1, &SearchFilter::default())
            .await
            .unwrap();
        assert_eq!(again.len(), 1);
        assert_eq!(again[0].document_id, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
