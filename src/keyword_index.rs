use std::collections::HashMap;
use std::sync::OnceLock;

use async_trait::async_trait;
use jieba_rs::Jieba;
use sqlx::{Executor, PgPool, Row};
use tokio_rusqlite::{Connection, params};

use crate::chunk::{ChunkMatch, ChunkRecord, SearchFilter};
use crate::error::BackendError;

/// Okapi BM25 调参常量。
const K1: f64 = 1.2;
const B: f64 = 0.75;

/// 关键词索引抽象，形状与 `VectorStore` 平行：`search` 入参是原始查询文本，
/// 内部做分词与 BM25 打分。实现为 SQLite 持久化索引（postings/tf/df 双层表），
/// 进程重启后索引仍在，无需从向量库重建。
#[async_trait]
pub(crate) trait KeywordIndex: Send + Sync {
    async fn upsert(&self, records: Vec<ChunkRecord>) -> Result<(), BackendError>;
    async fn remove_document(&self, document_id: u64) -> Result<(), BackendError>;
    async fn search(
        &self,
        query_text: &str,
        limit: usize,
        filter: &SearchFilter,
    ) -> Result<Vec<ChunkMatch>, BackendError>;
    /// 清空全部索引项。重置文档库时调用。
    async fn clear(&self) -> Result<(), BackendError>;
}

/// SQLite 持久化 BM25 索引：
/// - `chunk_meta`：每 chunk 的文本与 token 总数（BM25 的文档长度 dl）。
/// - `chunk_terms`：每个 chunk 的 term 频次表（postings），term 上有索引，查询按 term 走索引。
///
/// df（term 出现在多少个 chunk 中）由 `chunk_terms` 按 term 分组 COUNT 得出，不单独存储，
/// 保证增删改后统计自然一致，无需手工维护反链计数。
#[derive(Clone)]
pub(crate) struct SqliteBm25Index {
    conn: Connection,
}

const KEYWORD_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS chunk_meta (
    document_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    text        TEXT NOT NULL,
    PRIMARY KEY (document_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS chunk_terms (
    document_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    term        TEXT NOT NULL,
    tf          INTEGER NOT NULL,
    PRIMARY KEY (document_id, chunk_index, term)
);

CREATE INDEX IF NOT EXISTS idx_chunk_terms_term ON chunk_terms(term);
"#;

impl SqliteBm25Index {
    /// 从已有的 SQLite 连接创建索引（与 TaskStore 共用一个连接，避免并发写锁）。
    pub(crate) async fn from_conn(conn: Connection) -> Result<Self, BackendError> {
        conn.call(|conn| {
            conn.execute_batch(KEYWORD_SCHEMA)?;
            Ok(())
        })
        .await
        .map_err(keyword_err)?;
        Ok(Self { conn })
    }

    /// 内存索引，仅测试用。
    #[cfg(test)]
    pub(crate) async fn open_in_memory() -> Result<Self, BackendError> {
        let conn = Connection::open_in_memory()
            .await
            .map_err(|error| BackendError::keyword_index(error.to_string()))?;
        Self::from_conn(conn).await
    }
}

/// 从 postings 里查一批 term 的 df（term 出现的 chunk 数）。返回 term -> df。
fn fetch_df(conn: &rusqlite::Connection, terms: &[&str]) -> rusqlite::Result<HashMap<String, u32>> {
    if terms.is_empty() {
        return Ok(HashMap::new());
    }
    let placeholders = terms.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
    let sql = format!(
        "SELECT term, COUNT(*) FROM chunk_terms WHERE term IN ({placeholders}) GROUP BY term"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(terms), |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as u32))
    })?;
    let mut df = HashMap::new();
    for row in rows {
        let (term, count) = row?;
        df.insert(term, count);
    }
    Ok(df)
}

#[async_trait]
impl KeywordIndex for SqliteBm25Index {
    async fn upsert(&self, records: Vec<ChunkRecord>) -> Result<(), BackendError> {
        if records.is_empty() {
            return Ok(());
        }
        let conn = self.conn.clone();
        conn.call(move |conn| {
            let tx = conn.transaction()?;
            // 先整文档删除，再插入：chunk 集发生变化时旧 chunk 的 term 才不会残留。
            let affected: std::collections::HashSet<u64> =
                records.iter().map(|r| r.document_id).collect();
            for document_id in &affected {
                tx.execute(
                    "DELETE FROM chunk_terms WHERE document_id = ?1",
                    params![*document_id as i64],
                )?;
                tx.execute(
                    "DELETE FROM chunk_meta WHERE document_id = ?1",
                    params![*document_id as i64],
                )?;
            }
            for record in &records {
                tx.execute(
                    "INSERT OR REPLACE INTO chunk_meta (document_id, chunk_index, token_count, text)
                     VALUES (?1, ?2, ?3, ?4)",
                    params![record.document_id as i64, record.chunk_index as i64, 0i64, &record.text],
                )?;
                let tokens = tokenize(&record.text);
                tx.execute(
                    "UPDATE chunk_meta SET token_count=?1 WHERE document_id=?2 AND chunk_index=?3",
                    params![tokens.len() as i64, record.document_id as i64, record.chunk_index as i64],
                )?;
                let mut freqs: HashMap<String, u32> = HashMap::new();
                for term in &tokens {
                    *freqs.entry(term.clone()).or_default() += 1;
                }
                for (term, tf) in freqs {
                    tx.execute(
                        "INSERT OR REPLACE INTO chunk_terms (document_id, chunk_index, term, tf)
                         VALUES (?1, ?2, ?3, ?4)",
                        params![
                            record.document_id as i64,
                            record.chunk_index as i64,
                            term,
                            tf as i64
                        ],
                    )?;
                }
            }
            tx.commit()?;
            Ok(())
        })
        .await
        .map_err(keyword_err)?;
        Ok(())
    }

    async fn remove_document(&self, document_id: u64) -> Result<(), BackendError> {
        let conn = self.conn.clone();
        conn.call(move |conn| {
            let tx = conn.transaction()?;
            tx.execute(
                "DELETE FROM chunk_terms WHERE document_id = ?1",
                params![document_id as i64],
            )?;
            tx.execute(
                "DELETE FROM chunk_meta WHERE document_id = ?1",
                params![document_id as i64],
            )?;
            tx.commit()?;
            Ok(())
        })
        .await
        .map_err(keyword_err)?;
        Ok(())
    }

    async fn clear(&self) -> Result<(), BackendError> {
        let conn = self.conn.clone();
        conn.call(move |conn| {
            let tx = conn.transaction()?;
            tx.execute("DELETE FROM chunk_terms", [])?;
            tx.execute("DELETE FROM chunk_meta", [])?;
            tx.commit()?;
            Ok(())
        })
        .await
        .map_err(keyword_err)?;
        Ok(())
    }

    async fn search(
        &self,
        query_text: &str,
        limit: usize,
        filter: &SearchFilter,
    ) -> Result<Vec<ChunkMatch>, BackendError> {
        let query_terms = tokenize(query_text);
        if query_terms.is_empty() || filter.is_empty() {
            return Ok(Vec::new());
        }
        // 闭包要求 'static：把过滤条件转成属主值再移入。
        let document_ids: Option<Vec<u64>> = filter
            .document_ids
            .as_ref()
            .map(|ids| ids.iter().copied().collect());
        let conn = self.conn.clone();
        conn.call(move |conn| {
            let terms: Vec<String> = query_terms.to_vec();
            let unique_terms: Vec<String> = {
                let mut seen = std::collections::HashSet::new();
                terms
                    .into_iter()
                    .filter(|t| seen.insert(t.clone()))
                    .collect()
            };
            let refs: Vec<&str> = unique_terms.iter().map(String::as_str).collect();
            let df = fetch_df(conn, &refs)?;
            if df.is_empty() {
                return Ok(Ok(Vec::new()));
            }
            let (n, avgdl): (i64, f64) = conn.query_row(
                "SELECT COUNT(*), IFNULL(AVG(token_count), 0.0) FROM chunk_meta",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?;
            if n == 0 {
                return Ok(Ok(Vec::new()));
            }
            let avgdl = if avgdl > 0.0 { avgdl } else { 1.0 };
            let n = n as f64;

            // 取查询 term 的全部 postings，串接 chunk_meta 拿到 dl（token_count）与文本。
            let placeholders = unique_terms
                .iter()
                .map(|_| "?")
                .collect::<Vec<_>>()
                .join(", ");
            let mut sql = format!(
                "SELECT t.document_id, t.chunk_index, t.term, t.tf, m.token_count, m.text
                 FROM chunk_terms t JOIN chunk_meta m
                   ON m.document_id = t.document_id AND m.chunk_index = t.chunk_index
                 WHERE t.term IN ({placeholders})"
            );
            let mut bound: Vec<rusqlite::types::Value> = unique_terms
                .iter()
                .map(|t| rusqlite::types::Value::Text(t.clone()))
                .collect();
            if let Some(ids) = &document_ids {
                let id_placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
                sql.push_str(&format!(" AND t.document_id IN ({id_placeholders})"));
                bound.extend(
                    ids.iter()
                        .map(|&id| rusqlite::types::Value::Integer(id as i64)),
                );
            }
            sql.push_str(" ORDER BY t.document_id, t.chunk_index, t.term");

            let mut stmt = conn.prepare(&sql)?;
            // 每个 chunk 累加 BM25 分：score += idf * tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl))
            let mut scored: HashMap<(u64, usize), (f64, String)> = HashMap::new();
            struct Row {
                document_id: u64,
                chunk_index: u64,
                term: String,
                tf: f64,
                dl: f64,
                text: String,
            }
            let rows = stmt.query_map(rusqlite::params_from_iter(bound.iter()), |row| {
                Ok(Row {
                    document_id: row.get::<_, i64>(0)? as u64,
                    chunk_index: row.get::<_, i64>(1)? as u64,
                    term: row.get(2)?,
                    tf: row.get::<_, i64>(3)? as f64,
                    dl: row.get::<_, i64>(4)? as f64,
                    text: row.get(5)?,
                })
            })?;
            for row in rows {
                let row = row?;
                let term_df = *df.get(&row.term).unwrap_or(&0) as f64;
                if term_df == 0.0 {
                    continue;
                }
                // 加 1 平滑的 idf，避免 df > N/2 时出现负分。
                let idf = (1.0 + (n - term_df + 0.5) / (term_df + 0.5)).ln();
                let denom = row.tf + K1 * (1.0 - B + B * (row.dl / avgdl));
                let contrib = idf * (row.tf * (K1 + 1.0)) / denom;
                let entry = scored
                    .entry((row.document_id, row.chunk_index as usize))
                    .or_insert_with(|| (0.0, row.text.clone()));
                entry.0 += contrib;
            }
            let mut matches: Vec<ChunkMatch> = scored
                .into_iter()
                .filter(|(_, (score, _))| *score > 0.0)
                .map(|((doc, chunk), (score, text))| ChunkMatch {
                    document_id: doc,
                    chunk_index: chunk,
                    text,
                    score: score as f32,
                })
                .collect();
            matches.sort_by(|left, right| right.score.total_cmp(&left.score));
            matches.truncate(limit);
            Ok(Ok(matches))
        })
        .await
        .map_err(keyword_err)?
    }
}

/// PostgreSQL 持久化 BM25 索引：表结构与 SQLite 版一致（`chunk_meta` / `chunk_terms`），
/// 但 postings 落在 PG 中，可支撑数十万级 chunk 的倒排查询。分词复用 jieba（与 SQLite 版相同），
/// 因此对同一文本产出完全一致的 token 序列与 BM25 分数。
#[derive(Clone)]
pub(crate) struct PgBm25Index {
    pool: PgPool,
}

impl PgBm25Index {
    pub(crate) async fn new(database_url: &str) -> Result<Self, BackendError> {
        let pool = PgPool::connect(database_url)
            .await
            .map_err(|error| BackendError::keyword_index(error.to_string()))?;
        pool.execute(KEYWORD_SCHEMA_PG)
            .await
            .map_err(|error| BackendError::keyword_index(error.to_string()))?;
        Ok(Self { pool })
    }

    async fn execute(&self, sql: &str) -> Result<(), BackendError> {
        self.pool
            .execute(sql)
            .await
            .map(|_| ())
            .map_err(|error| BackendError::keyword_index(error.to_string()))
    }
}

/// PG 版表结构：与 SQLite 的 `KEYWORD_SCHEMA` 对齐，term 上加 btree 索引支撑 `IN (...)` 查询。
const KEYWORD_SCHEMA_PG: &str = r#"
CREATE TABLE IF NOT EXISTS chunk_meta (
    document_id BIGINT  NOT NULL,
    chunk_index INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    PRIMARY KEY (document_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS chunk_terms (
    document_id BIGINT  NOT NULL,
    chunk_index INTEGER NOT NULL,
    term        TEXT    NOT NULL,
    tf          INTEGER NOT NULL,
    PRIMARY KEY (document_id, chunk_index, term)
);

CREATE INDEX IF NOT EXISTS idx_chunk_terms_term ON chunk_terms(term);
"#;

#[async_trait]
impl KeywordIndex for PgBm25Index {
    async fn upsert(&self, records: Vec<ChunkRecord>) -> Result<(), BackendError> {
        if records.is_empty() {
            return Ok(());
        }
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|error| BackendError::keyword_index(error.to_string()))?;
        let affected: std::collections::HashSet<u64> =
            records.iter().map(|r| r.document_id).collect();
        for document_id in &affected {
            sqlx::query("DELETE FROM chunk_terms WHERE document_id = $1")
                .bind(*document_id as i64)
                .execute(&mut *tx)
                .await
                .map_err(|error| BackendError::keyword_index(error.to_string()))?;
            sqlx::query("DELETE FROM chunk_meta WHERE document_id = $1")
                .bind(*document_id as i64)
                .execute(&mut *tx)
                .await
                .map_err(|error| BackendError::keyword_index(error.to_string()))?;
        }
        for record in &records {
            let tokens = tokenize(&record.text);
            let token_count = tokens.len() as i32;
            sqlx::query(
                "INSERT INTO chunk_meta (document_id, chunk_index, token_count, text)
                 VALUES ($1, $2, $3, $4)",
            )
            .bind(record.document_id as i64)
            .bind(record.chunk_index as i32)
            .bind(token_count)
            .bind(&record.text)
            .execute(&mut *tx)
            .await
            .map_err(|error| BackendError::keyword_index(error.to_string()))?;
            let mut freqs: HashMap<String, u32> = HashMap::new();
            for term in &tokens {
                *freqs.entry(term.clone()).or_default() += 1;
            }
            for (term, tf) in freqs {
                sqlx::query(
                    "INSERT INTO chunk_terms (document_id, chunk_index, term, tf)
                     VALUES ($1, $2, $3, $4)",
                )
                .bind(record.document_id as i64)
                .bind(record.chunk_index as i32)
                .bind(&term)
                .bind(tf as i32)
                .execute(&mut *tx)
                .await
                .map_err(|error| BackendError::keyword_index(error.to_string()))?;
            }
        }
        tx.commit()
            .await
            .map_err(|error| BackendError::keyword_index(error.to_string()))?;
        Ok(())
    }

    async fn remove_document(&self, document_id: u64) -> Result<(), BackendError> {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|error| BackendError::keyword_index(error.to_string()))?;
        sqlx::query("DELETE FROM chunk_terms WHERE document_id = $1")
            .bind(document_id as i64)
            .execute(&mut *tx)
            .await
            .map_err(|error| BackendError::keyword_index(error.to_string()))?;
        sqlx::query("DELETE FROM chunk_meta WHERE document_id = $1")
            .bind(document_id as i64)
            .execute(&mut *tx)
            .await
            .map_err(|error| BackendError::keyword_index(error.to_string()))?;
        tx.commit()
            .await
            .map_err(|error| BackendError::keyword_index(error.to_string()))?;
        Ok(())
    }

    async fn clear(&self) -> Result<(), BackendError> {
        self.execute("DELETE FROM chunk_terms").await?;
        self.execute("DELETE FROM chunk_meta").await?;
        Ok(())
    }

    async fn search(
        &self,
        query_text: &str,
        limit: usize,
        filter: &SearchFilter,
    ) -> Result<Vec<ChunkMatch>, BackendError> {
        let query_terms = tokenize(query_text);
        if query_terms.is_empty() || filter.is_empty() {
            return Ok(Vec::new());
        }
        let mut unique_terms = Vec::new();
        {
            let mut seen = std::collections::HashSet::new();
            for term in query_terms {
                if seen.insert(term.clone()) {
                    unique_terms.push(term);
                }
            }
        }
        if unique_terms.is_empty() {
            return Ok(Vec::new());
        }

        // 下推 BM25 打分到 PG：df / n / avgdl 作为 CTE 在库内算好，命中面再大也只
        // 回传 ORDER BY score DESC LIMIT 的 top-k，避免把全量 postings 拉到应用层。
        // 打分公式与 SQLite 版保持一致（同一 K1/B 与平滑 idf）。
        let k = unique_terms.len();
        let score_expr = format!(
            "LN(1.0 + (s.n - df.df + 0.5) / (df.df + 0.5)) \
             * (t.tf::float8 * {k1p1:.3}) \
             / (t.tf::float8 + {k1:.3} * (1.0 - {b:.3} + {b:.3} * (m.token_count::float8 / s.avgdl)))",
            k1p1 = K1 + 1.0,
            k1 = K1,
            b = B,
        );
        let mut sql = String::from(
            "WITH stats AS (
                 SELECT COUNT(*)::float8 AS n,
                        COALESCE(AVG(token_count)::float8, 0.0) AS avgdl
                 FROM chunk_meta
             ),
             df AS (
                 SELECT term, COUNT(*)::float8 AS df
                 FROM chunk_terms
                 WHERE term IN (",
        );
        sql.push_str(
            &(0..k)
                .map(|i| format!("${}", i + 1))
                .collect::<Vec<_>>()
                .join(", "),
        );
        sql.push_str(
            ")
                 GROUP BY term
             )
             SELECT document_id, chunk_index, text, score FROM (
                 SELECT t.document_id, t.chunk_index, m.text,
                        SUM(",
        );
        sql.push_str(&score_expr);
        sql.push_str(
            ") AS score
                 FROM chunk_terms t
                 JOIN chunk_meta m
                   ON m.document_id = t.document_id AND m.chunk_index = t.chunk_index
                 JOIN df ON df.term = t.term
                 CROSS JOIN stats s
                 WHERE t.term IN (",
        );
        sql.push_str(
            &(0..k)
                .map(|i| format!("${}", i + 1))
                .collect::<Vec<_>>()
                .join(", "),
        );
        sql.push(')');
        let mut param_index = k;
        if let Some(ids) = &filter.document_ids {
            let mut id_list = Vec::new();
            for id in ids {
                id_list.push(id);
            }
            if !id_list.is_empty() {
                sql.push_str(" AND t.document_id IN (");
                let mut placeholders = Vec::new();
                for _ in &id_list {
                    param_index += 1;
                    placeholders.push(format!("${param_index}"));
                }
                sql.push_str(&placeholders.join(", "));
                sql.push(')');
            }
        }
        param_index += 1;
        sql.push_str(&format!(
            "                 GROUP BY t.document_id, t.chunk_index, m.text
             ) r WHERE r.score > 0.0 ORDER BY r.score DESC LIMIT ${param_index}"
        ));
        let mut builder = sqlx::query(&sql);
        for term in &unique_terms {
            builder = builder.bind(term);
        }
        if let Some(ids) = &filter.document_ids {
            for id in ids {
                builder = builder.bind(*id as i64);
            }
        }
        builder = builder.bind(limit as i64);
        let rows = builder
            .fetch_all(&self.pool)
            .await
            .map_err(|error| BackendError::keyword_index(error.to_string()))?;

        let mut matches: Vec<ChunkMatch> = Vec::with_capacity(rows.len());
        for row in rows {
            matches.push(ChunkMatch {
                document_id: row.get::<i64, _>("document_id") as u64,
                chunk_index: row.get::<i32, _>("chunk_index") as usize,
                text: row.get::<String, _>("text"),
                score: row.get::<f64, _>("score") as f32,
            });
        }
        Ok(matches)
    }
}

/// jieba 分词：精确模式（HMM），转小写，去空白。`Jieba` 实例全局缓存，词典只加载一次。
fn tokenize(text: &str) -> Vec<String> {
    static JIEBA: OnceLock<Jieba> = OnceLock::new();
    let jieba = JIEBA.get_or_init(Jieba::new);
    jieba
        .cut(text, false)
        .iter()
        .map(|token| token.to_lowercase())
        .filter(|token| !token.trim().is_empty())
        .collect()
}

fn keyword_err(error: tokio_rusqlite::Error) -> BackendError {
    eprintln!("keyword index error: {error}");
    BackendError::keyword_index(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keyword_index::KeywordIndex;

    fn record(document_id: u64, chunk_index: usize, text: &str) -> ChunkRecord {
        ChunkRecord {
            document_id,
            chunk_index,
            text: text.to_owned(),
            vector: Vec::new(),
        }
    }

    #[tokio::test]
    async fn bm25_ranks_relevant_chunk_first() {
        let index = SqliteBm25Index::open_in_memory().await.unwrap();
        index
            .upsert(vec![
                record(1, 0, "rust ownership and borrowing rules"),
                record(2, 0, "tokio async runtime scheduler"),
            ])
            .await
            .unwrap();
        let matches = index
            .search("ownership", 2, &SearchFilter::default())
            .await
            .unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].document_id, 1);
        assert!(matches[0].score > 0.0);
    }

    #[tokio::test]
    async fn bm25_chinese_tokenization() {
        let index = SqliteBm25Index::open_in_memory().await.unwrap();
        index
            .upsert(vec![record(
                1,
                0,
                "Rust 语言通过所有权机制保证内存安全与并发安全",
            )])
            .await
            .unwrap();
        let matches = index
            .search("内存安全", 1, &SearchFilter::default())
            .await
            .unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].document_id, 1);
    }

    #[tokio::test]
    async fn upsert_replaces_existing_chunks() {
        let index = SqliteBm25Index::open_in_memory().await.unwrap();
        index
            .upsert(vec![record(1, 0, "rust ownership")])
            .await
            .unwrap();
        // 重新 upsert 同一文档，旧 chunk 应被替换。
        index
            .upsert(vec![record(1, 0, "tokio runtime")])
            .await
            .unwrap();
        let matches = index
            .search("ownership", 1, &SearchFilter::default())
            .await
            .unwrap();
        assert!(matches.is_empty(), "old chunk should be replaced");
        let matches = index
            .search("tokio", 1, &SearchFilter::default())
            .await
            .unwrap();
        assert_eq!(matches.len(), 1);
    }

    #[tokio::test]
    async fn bm25_respects_document_filter() {
        let index = SqliteBm25Index::open_in_memory().await.unwrap();
        index
            .upsert(vec![
                record(1, 0, "rust ownership rules"),
                record(2, 0, "ownership in go differs"),
            ])
            .await
            .unwrap();
        let filter = SearchFilter {
            document_ids: Some(std::collections::HashSet::from([2])),
        };
        let matches = index.search("ownership", 10, &filter).await.unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].document_id, 2);
        // 空范围过滤器短路为空结果。
        let empty = SearchFilter {
            document_ids: Some(std::collections::HashSet::new()),
        };
        let matches = index.search("ownership", 10, &empty).await.unwrap();
        assert!(matches.is_empty());
    }
}
