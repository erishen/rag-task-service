use std::{
    collections::{HashMap, HashSet},
    num::NonZeroU32,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use governor::{Quota, RateLimiter};
use tokio::{
    sync::{broadcast, mpsc},
    task::yield_now,
};
use tokio_rusqlite::Connection;

use crate::{
    chunk::{ChunkMatch, ChunkRecord, SearchFilter},
    config::{Config, VectorStoreKind},
    embedding::{Embedder, HashEmbedder, OllamaEmbedder},
    error::ApiError,
    fusion::rrf_fuse,
    keyword_index::{KeywordIndex, PgBm25Index, SqliteBm25Index},
    middleware::KeyedLimiter,
    model::{
        CreateTaskRequest, DocumentImportResponse, DocumentPage, DocumentStatus, EventPage,
        EventQuery, Priority, RagDocument, SearchMode, SearchQuery, SearchResponse, SearchResult,
        Task, TaskEvent, TaskKind, TaskPage, TaskQuery, TaskStats, TaskStatus, UpdateTaskRequest,
    },
    rag::chunk_text,
    rerank::{LlmReranker, Reranker},
    vector_store::{InMemoryVectorStore, LanceDbVectorStore, PgVectorStore, VectorStore},
};

const DEFAULT_PAGE_SIZE: usize = 20;
const MAX_PAGE_SIZE: usize = 100;

/// 建表 + PRAGMA。WAL 提升并发读，events_trim 触发器维持事件环形缓冲（最新 200 条）。
const SCHEMA_SQL: &str = r#"
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    status      TEXT    NOT NULL,
    kind        TEXT    NOT NULL,
    priority    TEXT    NOT NULL,
    progress    INTEGER NOT NULL,
    detail      TEXT,
    version     INTEGER NOT NULL,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);

CREATE TABLE IF NOT EXISTS documents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    status        TEXT    NOT NULL,
    source_chars  INTEGER NOT NULL,
    chunk_count   INTEGER NOT NULL,
    error         TEXT,
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS document_contents (
    document_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    content     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT    NOT NULL,
    task_id      INTEGER NOT NULL,
    task_version INTEGER NOT NULL,
    occurred_at  INTEGER NOT NULL
);

CREATE TRIGGER IF NOT EXISTS events_trim
AFTER INSERT ON events
WHEN (SELECT COUNT(*) FROM events) > 200
BEGIN
    DELETE FROM events WHERE id = (SELECT id FROM events ORDER BY id ASC LIMIT 1);
END;
"#;

#[derive(Clone)]
pub struct AppState {
    pub(crate) store: Arc<TaskStore>,
    embedder: Arc<dyn Embedder>,
    vector_store: Arc<dyn VectorStore>,
    keyword_index: Arc<dyn KeywordIndex>,
    reranker: Option<Arc<dyn Reranker>>,
    pub(crate) config: Config,
    pub(crate) rate_limiter: Arc<KeyedLimiter>,
    import_tx: mpsc::Sender<ImportJob>,
    /// 事件广播通道：任务创建/更新/删除、RAG 导入阶段变更都经此推给 SSE 订阅者。
    pub(crate) event_tx: broadcast::Sender<TaskEvent>,
}

impl AppState {
    /// 根据 Config 构造 embedding、向量库、关键词索引与重排器。默认走 hash embedding +
    /// 内存向量库；配置了 `EMBEDDING_API_BASE` / `VECTOR_STORE=lancedb` 时切换到 Ollama + LanceDB。
    pub async fn new(config: Config) -> Self {
        let (event_tx, _) = broadcast::channel(256);
        let store = Arc::new(
            TaskStore::open(&config.db_path, event_tx.clone())
                .await
                .expect("failed to open sqlite store"),
        );
        let embedder: Arc<dyn Embedder> = if config.use_remote_embedding() {
            let base = config
                .embedding_api_base
                .as_deref()
                .expect("embedding api base is required when remote embedding is enabled");
            let embedder = OllamaEmbedder::new(
                base,
                config.embedding_api_key.clone(),
                config.embedding_model.clone(),
            )
            .await
            .expect("failed to initialize Ollama embedder; is `ollama serve` running?");
            Arc::new(embedder)
        } else {
            Arc::new(HashEmbedder::new())
        };
        let dim = embedder.dim();
        let vector_store: Arc<dyn VectorStore> = match config.vector_store {
            VectorStoreKind::LanceDb => {
                let store = LanceDbVectorStore::new(&config.lancedb_path, dim)
                    .await
                    .expect("failed to open LanceDB vector store");
                Arc::new(store)
            }
            VectorStoreKind::PgVector => {
                let database_url = config
                    .database_url
                    .clone()
                    .expect("DATABASE_URL is required when VECTOR_STORE=pgvector");
                let store = PgVectorStore::new(&database_url, dim)
                    .await
                    .expect("failed to open PostgreSQL pgvector store");
                Arc::new(store)
            }
            VectorStoreKind::InMemory => Arc::new(InMemoryVectorStore::new()),
        };
        // 关键词索引：PG 模式下与 pgvector 同库（`chunk_meta`/`chunk_terms` 进 PG），
        // SQLite/LanceDB 模式仍共享 TaskStore 的 SQLite 连接（避免第二个写连接竞争锁）。
        let keyword_index: Arc<dyn KeywordIndex> = match config.vector_store {
            VectorStoreKind::PgVector => {
                let database_url = config
                    .database_url
                    .clone()
                    .expect("DATABASE_URL is required when VECTOR_STORE=pgvector");
                Arc::new(
                    PgBm25Index::new(&database_url)
                        .await
                        .expect("failed to initialize postgres keyword index"),
                )
            }
            _ => Arc::new(
                SqliteBm25Index::from_conn(store.conn().clone())
                    .await
                    .expect("failed to initialize sqlite keyword index"),
            ),
        };
        let reranker: Option<Arc<dyn Reranker>> = if config.use_rerank() {
            Some(Arc::new(LlmReranker::new(
                config.rerank_api_base.as_deref().unwrap(),
                config.embedding_api_key.clone(),
                config.rerank_model.clone().unwrap(),
                config.rerank_top_n,
            )))
        } else {
            None
        };
        let rate_limiter = Arc::new(RateLimiter::keyed(Quota::per_second(
            NonZeroU32::new(config.rate_limit_per_sec.max(1)).expect("rate limit > 0"),
        )));
        let (import_tx, import_rx) = mpsc::channel(16);
        tokio::spawn(import_worker(
            store.clone(),
            embedder.clone(),
            vector_store.clone(),
            keyword_index.clone(),
            import_rx,
        ));
        Self {
            store,
            embedder,
            vector_store,
            keyword_index,
            reranker,
            config,
            rate_limiter,
            import_tx,
            event_tx,
        }
    }

    pub(crate) async fn enqueue_import(&self, job: ImportJob) -> Result<(), ApiError> {
        self.import_tx
            .send(job)
            .await
            .map_err(|_| ApiError::internal("document worker is unavailable"))
    }

    /// 重置 RAG 数据：清空文档/文档导入任务/相关事件 + 向量库 + BM25 索引。
    pub(crate) async fn reset(&self) -> Result<(), ApiError> {
        self.store.clear_documents().await?;
        self.vector_store.clear().await?;
        self.keyword_index.clear().await?;
        Ok(())
    }

    /// 删除单个文档：DB 元数据 + 向量库 chunks + BM25 索引。
    pub(crate) async fn delete_document(&self, id: u64) -> Result<(), ApiError> {
        self.store.delete_document(id).await?;
        self.vector_store.remove_document(id).await?;
        self.keyword_index.remove_document(id).await?;
        Ok(())
    }

    /// 检索编排：支持纯向量 / BM25+向量 RRF 混合，可选 LLM 重排。
    /// embedding、向量检索、关键词检索都不持锁，仅最后短暂查 SQLite 补全文档名。
    pub(crate) async fn search(&self, query: SearchQuery) -> Result<SearchResponse, ApiError> {
        let query_text = query.q.trim().to_owned();
        if query_text.is_empty() {
            return Err(ApiError::bad_request("query cannot be empty"));
        }
        let limit = page_size(query.limit);
        let mode = query.mode.unwrap_or(self.config.search_mode);
        let want_rerank =
            query.rerank.unwrap_or(self.config.use_rerank()) && self.reranker.is_some();
        // 开启重排时多召回一个候选池，否则直接取 limit。
        let pool = if want_rerank {
            self.config.rerank_top_n.max(limit)
        } else {
            limit
        };

        let mut vectors = self.embedder.embed(&[query_text.as_str()]).await?;
        let query_vector = vectors
            .pop()
            .ok_or_else(|| ApiError::internal("embedding returned no vector for query"))?;
        let filter = SearchFilter {
            document_ids: parse_document_filter(query.documents.as_deref())?,
        };

        let mut matches = match mode {
            SearchMode::Vector => {
                let mut found = self
                    .vector_store
                    .search(&query_vector, pool, &filter)
                    .await?;
                if want_rerank && let Some(reranker) = &self.reranker {
                    found = reranker.rerank(&query_text, found, limit).await?;
                }
                found
            }
            SearchMode::Hybrid => {
                // 两路并发检索，延迟取较慢者。过滤条件两路必须一致，RRF 才不串径。
                let (vector_matches, keyword_matches) = tokio::join!(
                    self.vector_store.search(&query_vector, pool, &filter),
                    self.keyword_index.search(&query_text, pool, &filter),
                );
                let vector_matches = vector_matches?;
                let keyword_matches = keyword_matches?;
                let k = self.config.rrf_k;
                let mut fused = rrf_fuse(vector_matches, keyword_matches, pool, k);
                // RRF 原始分很小（k=60 时双路最高 2/61），归一化到 [0,1] 保持前端百分比语义。
                let max_possible = 2.0_f32 / (k as f32 + 1.0);
                if max_possible > 0.0 {
                    for m in &mut fused {
                        m.score /= max_possible;
                    }
                }
                if want_rerank && let Some(reranker) = &self.reranker {
                    fused = reranker.rerank(&query_text, fused, limit).await?;
                }
                fused
            }
        };

        if matches.len() > limit {
            matches.truncate(limit);
        }
        let items = self.store.enrich_search(matches).await?;
        Ok(SearchResponse {
            query: query_text,
            items,
        })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ImportJob {
    document_id: u64,
    task_id: u64,
}

/// 持久化任务/文档/事件存储。单 `tokio_rusqlite::Connection`（内部 Arc 可廉价 Clone），
/// 所有 SQL 经 `conn.call` 串行化，等价于原来的 `Arc<RwLock<TaskStore>>` 但数据落盘。
#[derive(Clone)]
pub(crate) struct TaskStore {
    conn: Connection,
    /// 事件广播通道，DB 写操作落库后把事件推给 SSE 订阅者（无订阅者时静默忽略）。
    event_tx: broadcast::Sender<TaskEvent>,
    /// 文档 id -> 名称 的进程内缓存，避免每次检索都查 SQLite 补名。随增删改同步维护。
    document_names: Arc<std::sync::RwLock<HashMap<u64, String>>>,
}

impl TaskStore {
    /// 打开文件 DB 并建表，父目录不存在则创建。
    pub(crate) async fn open(
        path: &str,
        event_tx: broadcast::Sender<TaskEvent>,
    ) -> Result<Self, ApiError> {
        if let Some(parent) = std::path::Path::new(path).parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .map_err(|e| ApiError::internal(format!("create db dir failed: {e}")))?;
        }
        let conn = Connection::open(path).await.map_err(map_sqlite_err)?;
        conn.call(|conn| {
            conn.execute_batch(SCHEMA_SQL)?;
            Ok(())
        })
        .await
        .map_err(map_tokio_err)?;
        Ok(Self {
            conn,
            event_tx,
            document_names: Arc::new(std::sync::RwLock::new(HashMap::new())),
        })
    }

    /// 内存 DB，仅测试用，无文件依赖。事件广播通道仅供记录，无订阅者也能用。
    #[cfg(test)]
    pub(crate) async fn open_in_memory() -> Result<Self, ApiError> {
        let conn = Connection::open_in_memory().await.map_err(map_sqlite_err)?;
        conn.call(|conn| {
            conn.execute_batch(SCHEMA_SQL)?;
            Ok(())
        })
        .await
        .map_err(map_tokio_err)?;
        let (event_tx, _) = broadcast::channel(64);
        Ok(Self {
            conn,
            event_tx,
            document_names: Arc::new(std::sync::RwLock::new(HashMap::new())),
        })
    }

    /// 统一调用入口：闭包返回 `Result<Result<R, ApiError>, rusqlite::Error>`，
    /// 外层 rusqlite/传输错误由 `call_db` 转 ApiError，内层 ApiError 原样返回给调用方。
    /// 调用方拿到 `Result<R, ApiError>` 再用 `?` 解开业务错误。
    async fn call_db<F, R>(&self, function: F) -> Result<R, ApiError>
    where
        F: FnOnce(&mut rusqlite::Connection) -> Result<Result<R, ApiError>, rusqlite::Error>
            + Send
            + 'static,
        R: Send + 'static,
    {
        self.conn.call(function).await.map_err(map_tokio_err)?
    }

    /// 给关键词索引等共享同一连接的组件复用底层的 SQLite 连接句柄。
    pub(crate) fn conn(&self) -> &Connection {
        &self.conn
    }

    pub(crate) async fn create(&self, request: CreateTaskRequest) -> Result<Task, ApiError> {
        let title = validate_title(request.title)?;
        let priority = request.priority.unwrap_or_default();
        let event_tx = self.event_tx.clone();
        self.call_db(move |conn| {
            let task = create_task_inner(conn, &event_tx, title, priority, TaskKind::Manual, None)?;
            Ok(Ok(task))
        })
        .await
    }

    pub(crate) async fn queue_document(
        &self,
        name: String,
        content: String,
    ) -> Result<(DocumentImportResponse, ImportJob), ApiError> {
        let name = validate_document_name(name)?;
        if content.trim().is_empty() {
            return Err(ApiError::bad_request("document has no text content"));
        }
        let source_chars = content.chars().count() as i64;
        let event_tx = self.event_tx.clone();
        let (response, job) = self
            .call_db(move |conn| {
                let now = now_unix_seconds() as i64;
                conn.execute(
                    "INSERT INTO documents (name, status, source_chars, chunk_count, error, created_at, updated_at)
                     VALUES (?1, ?2, ?3, 0, NULL, ?4, ?4)",
                    rusqlite::params![&name, DocumentStatus::Queued.as_str(), source_chars, now],
                )?;
                let document_id = conn.last_insert_rowid() as u64;
                conn.execute(
                    "INSERT INTO document_contents (document_id, content) VALUES (?1, ?2)",
                    rusqlite::params![document_id as i64, &content],
                )?;
                let task = create_task_inner(
                    conn,
                    &event_tx,
                    format!("导入文档：{name}"),
                    Priority::High,
                    TaskKind::DocumentImport,
                    Some("等待 RAG Worker 处理".to_owned()),
                )?;
                let document = fetch_document(conn, document_id)?;
                let job = ImportJob {
                    document_id,
                    task_id: task.id,
                };
                Ok(Ok((DocumentImportResponse { document, task }, job)))
            })
            .await?;
        if let Ok(mut names) = self.document_names.write() {
            names.insert(response.document.id, response.document.name.clone());
        }
        Ok((response, job))
    }

    pub(crate) async fn clear_documents(&self) -> Result<(), ApiError> {
        self.call_db(move |conn| {
            // 先删文档导入任务的事件，再删任务，避免 orphan 事件
            conn.execute(
                "DELETE FROM events WHERE task_id IN (SELECT id FROM tasks WHERE kind = ?1)",
                rusqlite::params![TaskKind::DocumentImport.as_str()],
            )?;
            conn.execute(
                "DELETE FROM tasks WHERE kind = ?1",
                rusqlite::params![TaskKind::DocumentImport.as_str()],
            )?;
            conn.execute("DELETE FROM document_contents", [])?;
            conn.execute("DELETE FROM documents", [])?;
            Ok(Ok(()))
        })
        .await?;
        self.document_names
            .write()
            .map_err(|e| ApiError::internal(e.to_string()))?
            .clear();
        Ok(())
    }

    pub(crate) async fn delete_document(&self, id: u64) -> Result<(), ApiError> {
        if let Ok(mut names) = self.document_names.write() {
            names.remove(&id);
        }
        self.call_db(move |conn| {
            let affected = conn.execute(
                "DELETE FROM documents WHERE id = ?1",
                rusqlite::params![id as i64],
            )?;
            if affected == 0 {
                return Ok(Err(ApiError::not_found(format!(
                    "document {id} does not exist"
                ))));
            }
            // document_contents 靠 ON DELETE CASCADE 清理
            Ok(Ok(()))
        })
        .await
    }

    pub(crate) async fn get(&self, id: u64) -> Result<Task, ApiError> {
        self.call_db(move |conn| match fetch_task(conn, id) {
            Ok(task) => Ok(Ok(task)),
            Err(_) => Ok(Err(ApiError::not_found(format!(
                "task {id} does not exist"
            )))),
        })
        .await
    }

    pub(crate) async fn list(&self, query: TaskQuery) -> Result<TaskPage, ApiError> {
        let offset = query.offset.unwrap_or(0) as i64;
        let limit = page_size(query.limit) as i64;
        let status = query.status;
        self.call_db(move |conn| {
            let (total, items) = match status {
                Some(st) => {
                    let st_str = st.as_str();
                    let total: i64 = conn.query_row(
                        "SELECT COUNT(*) FROM tasks WHERE status = ?1",
                        rusqlite::params![st_str],
                        |row| row.get(0),
                    )?;
                    let mut stmt = conn.prepare(
                        "SELECT id, title, status, kind, priority, progress, detail, version, created_at, updated_at
                         FROM tasks WHERE status = ?1 ORDER BY id ASC LIMIT ?2 OFFSET ?3",
                    )?;
                    let items: Vec<Task> = stmt
                        .query_map(rusqlite::params![st_str, limit, offset], task_from_row)?
                        .collect::<rusqlite::Result<_>>()?;
                    (total, items)
                }
                None => {
                    let total: i64 =
                        conn.query_row("SELECT COUNT(*) FROM tasks", [], |row| row.get(0))?;
                    let mut stmt = conn.prepare(
                        "SELECT id, title, status, kind, priority, progress, detail, version, created_at, updated_at
                         FROM tasks ORDER BY id ASC LIMIT ?1 OFFSET ?2",
                    )?;
                    let items: Vec<Task> = stmt
                        .query_map(rusqlite::params![limit, offset], task_from_row)?
                        .collect::<rusqlite::Result<_>>()?;
                    (total, items)
                }
            };
            let total = total as usize;
            let fetched = items.len();
            let next_offset = (offset as usize + fetched < total).then_some(offset as usize + fetched);
            Ok(Ok(TaskPage {
                items,
                total,
                next_offset,
            }))
        })
        .await
    }

    pub(crate) async fn update(
        &self,
        id: u64,
        request: UpdateTaskRequest,
    ) -> Result<Task, ApiError> {
        if request.title.is_none() && request.status.is_none() && request.priority.is_none() {
            return Err(ApiError::bad_request(
                "provide at least one field to update",
            ));
        }
        let validated_title = match request.title {
            Some(t) => Some(validate_title(t)?),
            None => None,
        };
        let status_opt = request.status;
        let priority_opt = request.priority;
        let expected_version = request.expected_version;
        let event_tx = self.event_tx.clone();
        self.call_db(move |conn| {
            let mut task = match fetch_task(conn, id) {
                Ok(t) => t,
                Err(_) => {
                    return Ok(Err(ApiError::not_found(format!(
                        "task {id} does not exist"
                    ))));
                }
            };
            if task.version != expected_version {
                return Ok(Err(ApiError::conflict(format!(
                    "task {id} is at version {}; retry with that version",
                    task.version
                ))));
            }
            if let Some(t) = &validated_title {
                task.title = t.clone();
            }
            if let Some(s) = status_opt {
                task.status = s;
                task.progress = progress_for(s);
            }
            if let Some(p) = priority_opt {
                task.priority = p;
            }
            task.version += 1;
            task.updated_at = now_unix_seconds();
            conn.execute(
                "UPDATE tasks SET title=?1, status=?2, kind=?3, priority=?4, progress=?5,
                                   detail=?6, version=?7, created_at=?8, updated_at=?9
                 WHERE id=?10",
                rusqlite::params![
                    &task.title,
                    task.status.as_str(),
                    task.kind.as_str(),
                    task.priority.as_str(),
                    task.progress,
                    &task.detail,
                    task.version,
                    task.created_at,
                    task.updated_at,
                    id as i64
                ],
            )?;
            record_event(conn, &event_tx, "task.updated", &task)?;
            Ok(Ok(task))
        })
        .await
    }

    pub(crate) async fn delete(&self, id: u64, expected_version: u64) -> Result<(), ApiError> {
        let event_tx = self.event_tx.clone();
        self.call_db(move |conn| {
            let task = match fetch_task(conn, id) {
                Ok(t) => t,
                Err(_) => {
                    return Ok(Err(ApiError::not_found(format!(
                        "task {id} does not exist"
                    ))));
                }
            };
            if task.version != expected_version {
                return Ok(Err(ApiError::conflict(format!(
                    "task {id} is at version {}; retry with that version",
                    task.version
                ))));
            }
            conn.execute(
                "DELETE FROM tasks WHERE id=?1",
                rusqlite::params![id as i64],
            )?;
            record_event(conn, &event_tx, "task.deleted", &task)?;
            Ok(Ok(()))
        })
        .await
    }

    pub(crate) async fn events(&self, query: EventQuery) -> Result<EventPage, ApiError> {
        let after = query.after.unwrap_or(0) as i64;
        let limit = page_size(query.limit) as i64;
        self.call_db(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id, kind, task_id, task_version, occurred_at
                 FROM events WHERE id > ?1 ORDER BY id ASC LIMIT ?2",
            )?;
            let items: Vec<TaskEvent> = stmt
                .query_map(rusqlite::params![after, limit], |row| {
                    Ok(TaskEvent {
                        id: row.get::<_, i64>(0)? as u64,
                        kind: row.get(1)?,
                        task_id: row.get::<_, i64>(2)? as u64,
                        task_version: row.get::<_, i64>(3)? as u64,
                        occurred_at: row.get::<_, i64>(4)? as u64,
                    })
                })?
                .collect::<rusqlite::Result<_>>()?;
            let latest: i64 =
                conn.query_row("SELECT COALESCE(MAX(id), 0) FROM events", [], |row| {
                    row.get(0)
                })?;
            Ok(Ok(EventPage {
                items,
                latest_event_id: latest as u64,
            }))
        })
        .await
    }

    /// 取最近 `limit` 条事件（按时间正序），供 SSE 连接建立时回放历史快照。
    pub(crate) async fn recent_events(&self, limit: usize) -> Result<Vec<TaskEvent>, ApiError> {
        let limit = limit.clamp(1, 100) as i64;
        self.call_db(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id, kind, task_id, task_version, occurred_at
                 FROM (SELECT * FROM events ORDER BY id DESC LIMIT ?1)
                 ORDER BY id ASC",
            )?;
            let items: Vec<TaskEvent> = stmt
                .query_map(rusqlite::params![limit], |row| {
                    Ok(TaskEvent {
                        id: row.get::<_, i64>(0)? as u64,
                        kind: row.get(1)?,
                        task_id: row.get::<_, i64>(2)? as u64,
                        task_version: row.get::<_, i64>(3)? as u64,
                        occurred_at: row.get::<_, i64>(4)? as u64,
                    })
                })?
                .collect::<rusqlite::Result<_>>()?;
            Ok(Ok(items))
        })
        .await
    }

    /// 取 `after_id` 之后（不含）的全部事件，供 SSE 按 `Last-Event-ID` 续传。
    /// 事件表由触发器裁剪到最近 200 条，无 LIMIT 单趟至多扫 200 行，开销可忽略。
    pub(crate) async fn events_after(&self, after_id: u64) -> Result<Vec<TaskEvent>, ApiError> {
        self.call_db(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id, kind, task_id, task_version, occurred_at
                 FROM events WHERE id > ?1 ORDER BY id ASC",
            )?;
            let items: Vec<TaskEvent> = stmt
                .query_map(rusqlite::params![after_id as i64], |row| {
                    Ok(TaskEvent {
                        id: row.get::<_, i64>(0)? as u64,
                        kind: row.get(1)?,
                        task_id: row.get::<_, i64>(2)? as u64,
                        task_version: row.get::<_, i64>(3)? as u64,
                        occurred_at: row.get::<_, i64>(4)? as u64,
                    })
                })?
                .collect::<rusqlite::Result<_>>()?;
            Ok(Ok(items))
        })
        .await
    }

    pub(crate) async fn stats(&self) -> Result<TaskStats, ApiError> {
        self.call_db(|conn| {
            let stats = conn.query_row(
                "SELECT
                    COUNT(*),
                    COALESCE(SUM(CASE WHEN status='todo' THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN status='in_progress' THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN status='done' THEN 1 ELSE 0 END), 0),
                    (SELECT COUNT(*) FROM events)
                 FROM tasks",
                [],
                |row| {
                    Ok(TaskStats {
                        total: row.get::<_, i64>(0)? as usize,
                        todo: row.get::<_, i64>(1)? as usize,
                        in_progress: row.get::<_, i64>(2)? as usize,
                        done: row.get::<_, i64>(3)? as usize,
                        retained_events: row.get::<_, i64>(4)? as usize,
                    })
                },
            )?;
            Ok(Ok(stats))
        })
        .await
    }

    pub(crate) async fn task_count(&self) -> Result<usize, ApiError> {
        self.call_db(|conn| {
            let count: i64 = conn.query_row("SELECT COUNT(*) FROM tasks", [], |row| row.get(0))?;
            Ok(Ok(count as usize))
        })
        .await
    }

    pub(crate) async fn documents(&self) -> Result<DocumentPage, ApiError> {
        let page = self
            .call_db(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT id, name, status, source_chars, chunk_count, error, created_at, updated_at
                     FROM documents ORDER BY id ASC",
                )?;
                let items: Vec<RagDocument> = stmt
                    .query_map([], doc_from_row)?
                    .collect::<rusqlite::Result<_>>()?;
                Ok(Ok(DocumentPage { items }))
            })
            .await?;
        // 顺带把文档名缓存对齐，令搜索补名走缓存。
        if let Ok(mut names) = self.document_names.write() {
            names.clear();
            for doc in &page.items {
                names.insert(doc.id, doc.name.clone());
            }
        }
        Ok(page)
    }

    /// 把向量库返回的命中结果补全文档名，组装为对外 SearchResult。
    /// 文档名命中进程内缓存；缓存缺失的 id 才查 SQLite 并回填缓存。
    pub(crate) async fn enrich_search(
        &self,
        matches: Vec<ChunkMatch>,
    ) -> Result<Vec<SearchResult>, ApiError> {
        if matches.is_empty() {
            return Ok(Vec::new());
        }
        // 先从缓存取名字，收集缺失 id。
        let (mut name_map, missing) = {
            let cached = self
                .document_names
                .read()
                .map_err(|e| ApiError::internal(e.to_string()))?;
            let mut cached_name: HashMap<u64, String> = HashMap::new();
            let mut missing: Vec<u64> = Vec::new();
            for m in &matches {
                if let Some(name) = cached.get(&m.document_id) {
                    cached_name.insert(m.document_id, name.clone());
                } else {
                    missing.push(m.document_id);
                }
            }
            (cached_name, missing)
        };
        if !missing.is_empty() {
            let ids: Vec<i64> = missing.iter().map(|&id| id as i64).collect();
            self.call_db(move |conn| {
                let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                let sql = format!("SELECT id, name FROM documents WHERE id IN ({placeholders})");
                let mut stmt = conn.prepare(&sql)?;
                let rows: Vec<(u64, String)> = stmt
                    .query_map(rusqlite::params_from_iter(ids.iter()), |row| {
                        Ok((row.get::<_, i64>(0)? as u64, row.get::<_, String>(1)?))
                    })?
                    .collect::<rusqlite::Result<_>>()?;
                Ok(Ok(rows))
            })
            .await?
            .into_iter()
            .for_each(|(id, name)| {
                name_map.insert(id, name.clone());
            });
            if let Ok(mut names) = self.document_names.write() {
                names.extend(name_map.iter().map(|(id, name)| (*id, name.clone())));
            }
        }
        let items = matches
            .into_iter()
            .map(|m| SearchResult {
                document_name: name_map.get(&m.document_id).cloned().unwrap_or_default(),
                document_id: m.document_id,
                chunk_index: m.chunk_index,
                text: m.text,
                score: m.score,
            })
            .collect();
        Ok(items)
    }

    pub(crate) async fn begin_import(&self, job: ImportJob) -> Result<String, ApiError> {
        let doc_id = job.document_id;
        let task_id = job.task_id;
        let event_tx = self.event_tx.clone();
        self.call_db(move |conn| {
            let now = now_unix_seconds() as i64;
            let affected = conn.execute(
                "UPDATE documents SET status=?1, updated_at=?2 WHERE id=?3",
                rusqlite::params![DocumentStatus::Processing.as_str(), now, doc_id as i64],
            )?;
            if affected == 0 {
                return Ok(Err(ApiError::internal(
                    "document disappeared before processing",
                )));
            }
            let content: String = conn.query_row(
                "SELECT content FROM document_contents WHERE document_id=?1",
                rusqlite::params![doc_id as i64],
                |row| row.get(0),
            )?;
            set_import_task(
                conn,
                &event_tx,
                task_id,
                TaskStatus::InProgress,
                5,
                "解析文档内容",
                "rag.parsing",
            )?;
            Ok(Ok(content))
        })
        .await
    }

    pub(crate) async fn update_import_progress(
        &self,
        task_id: u64,
        progress: u8,
        detail: String,
    ) -> Result<(), ApiError> {
        let event_tx = self.event_tx.clone();
        self.call_db(move |conn| {
            set_import_task(
                conn,
                &event_tx,
                task_id,
                TaskStatus::InProgress,
                progress,
                &detail,
                "rag.embedding",
            )?;
            Ok(Ok(()))
        })
        .await
    }

    pub(crate) async fn complete_import(
        &self,
        job: ImportJob,
        chunk_count: usize,
    ) -> Result<(), ApiError> {
        let doc_id = job.document_id;
        let task_id = job.task_id;
        let chunks = chunk_count as i64;
        let event_tx = self.event_tx.clone();
        self.call_db(move |conn| {
            let now = now_unix_seconds() as i64;
            conn.execute(
                "UPDATE documents SET status=?1, chunk_count=?2, updated_at=?3 WHERE id=?4",
                rusqlite::params![DocumentStatus::Ready.as_str(), chunks, now, doc_id as i64],
            )?;
            set_import_task(
                conn,
                &event_tx,
                task_id,
                TaskStatus::Done,
                100,
                "已完成索引，可开始检索",
                "rag.indexed",
            )?;
            Ok(Ok(()))
        })
        .await
    }

    pub(crate) async fn fail_import(&self, job: ImportJob, error: String) -> Result<(), ApiError> {
        let doc_id = job.document_id;
        let task_id = job.task_id;
        let event_tx = self.event_tx.clone();
        self.call_db(move |conn| {
            let now = now_unix_seconds() as i64;
            conn.execute(
                "UPDATE documents SET status=?1, error=?2, updated_at=?3 WHERE id=?4",
                rusqlite::params![DocumentStatus::Failed.as_str(), &error, now, doc_id as i64],
            )?;
            let detail = format!("导入失败：{error}");
            set_import_task(
                conn,
                &event_tx,
                task_id,
                TaskStatus::Done,
                100,
                &detail,
                "rag.failed",
            )?;
            Ok(Ok(()))
        })
        .await
    }
}

// ---------- 闭包内同步辅助函数（接收 &mut rusqlite::Connection）----------

fn map_sqlite_err(error: rusqlite::Error) -> ApiError {
    eprintln!("sqlite error: {error}");
    ApiError::internal("database error")
}

/// `conn.call().await` 返回 `tokio_rusqlite::Error`（内部包装 rusqlite::Error + 连接级错误）。
fn map_tokio_err(error: tokio_rusqlite::Error) -> ApiError {
    eprintln!("tokio_rusqlite error: {error}");
    ApiError::internal("database error")
}

fn task_from_row(row: &rusqlite::Row) -> rusqlite::Result<Task> {
    let status_str: String = row.get(2)?;
    let kind_str: String = row.get(3)?;
    let priority_str: String = row.get(4)?;
    Ok(Task {
        id: row.get::<_, i64>(0)? as u64,
        title: row.get(1)?,
        status: parse_status(&status_str, 2)?,
        kind: parse_kind(&kind_str, 3)?,
        priority: parse_priority(&priority_str, 4)?,
        progress: row.get::<_, i64>(5)? as u8,
        detail: row.get(6)?,
        version: row.get::<_, i64>(7)? as u64,
        created_at: row.get::<_, i64>(8)? as u64,
        updated_at: row.get::<_, i64>(9)? as u64,
    })
}

fn doc_from_row(row: &rusqlite::Row) -> rusqlite::Result<RagDocument> {
    let status_str: String = row.get(2)?;
    Ok(RagDocument {
        id: row.get::<_, i64>(0)? as u64,
        name: row.get(1)?,
        status: parse_doc_status(&status_str, 2)?,
        source_chars: row.get::<_, i64>(3)? as usize,
        chunk_count: row.get::<_, i64>(4)? as usize,
        error: row.get(5)?,
        created_at: row.get::<_, i64>(6)? as u64,
        updated_at: row.get::<_, i64>(7)? as u64,
    })
}

fn fetch_task(conn: &rusqlite::Connection, id: u64) -> rusqlite::Result<Task> {
    conn.query_row(
        "SELECT id, title, status, kind, priority, progress, detail, version, created_at, updated_at
         FROM tasks WHERE id = ?1",
        rusqlite::params![id as i64],
        task_from_row,
    )
}

fn fetch_document(conn: &rusqlite::Connection, id: u64) -> rusqlite::Result<RagDocument> {
    conn.query_row(
        "SELECT id, name, status, source_chars, chunk_count, error, created_at, updated_at
         FROM documents WHERE id = ?1",
        rusqlite::params![id as i64],
        doc_from_row,
    )
}

/// 创建任务并记录 `task.created` 事件。
fn create_task_inner(
    conn: &mut rusqlite::Connection,
    event_tx: &broadcast::Sender<TaskEvent>,
    title: String,
    priority: Priority,
    kind: TaskKind,
    detail: Option<String>,
) -> rusqlite::Result<Task> {
    let now = now_unix_seconds() as i64;
    conn.execute(
        "INSERT INTO tasks (title, status, kind, priority, progress, detail, version, created_at, updated_at)
         VALUES (?1, 'todo', ?2, ?3, 0, ?4, 1, ?5, ?5)",
        rusqlite::params![&title, kind.as_str(), priority.as_str(), &detail, now],
    )?;
    let task = Task {
        id: conn.last_insert_rowid() as u64,
        title,
        status: TaskStatus::Todo,
        kind,
        priority,
        progress: 0,
        detail,
        version: 1,
        created_at: now as u64,
        updated_at: now as u64,
    };
    record_event(conn, event_tx, "task.created", &task)?;
    Ok(task)
}

/// 更新导入任务状态并记录事件。任务不存在视为错误（调用方应保证 job.task_id 有效）。
fn set_import_task(
    conn: &mut rusqlite::Connection,
    event_tx: &broadcast::Sender<TaskEvent>,
    task_id: u64,
    status: TaskStatus,
    progress: u8,
    detail: &str,
    event: &str,
) -> rusqlite::Result<()> {
    let now = now_unix_seconds() as i64;
    let affected = conn.execute(
        "UPDATE tasks SET status=?1, progress=?2, detail=?3, version=version+1, updated_at=?4
         WHERE id=?5",
        rusqlite::params![status.as_str(), progress, detail, now, task_id as i64],
    )?;
    if affected == 0 {
        return Err(rusqlite::Error::QueryReturnedNoRows);
    }
    let task = fetch_task(conn, task_id)?;
    record_event(conn, event_tx, event, &task)?;
    Ok(())
}

fn record_event(
    conn: &mut rusqlite::Connection,
    event_tx: &broadcast::Sender<TaskEvent>,
    kind: &str,
    task: &Task,
) -> rusqlite::Result<()> {
    let occurred_at = now_unix_seconds();
    conn.execute(
        "INSERT INTO events (kind, task_id, task_version, occurred_at) VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![
            kind,
            task.id as i64,
            task.version as i64,
            occurred_at as i64
        ],
    )?;
    // events_trim 触发器自动维护 200 条上限。
    // 推给 SSE 订阅者；无订阅者时 send 失败，事件已持久化，静默忽略即可。
    let event = TaskEvent {
        id: conn.last_insert_rowid() as u64,
        kind: kind.to_owned(),
        task_id: task.id,
        task_version: task.version,
        occurred_at,
    };
    let _ = event_tx.send(event);
    Ok(())
}

fn parse_status(s: &str, col: usize) -> rusqlite::Result<TaskStatus> {
    TaskStatus::parse(s).ok_or_else(|| sql_enum_error(col, "status", s))
}

fn parse_kind(s: &str, col: usize) -> rusqlite::Result<TaskKind> {
    TaskKind::parse(s).ok_or_else(|| sql_enum_error(col, "kind", s))
}

fn parse_priority(s: &str, col: usize) -> rusqlite::Result<Priority> {
    Priority::parse(s).ok_or_else(|| sql_enum_error(col, "priority", s))
}

fn parse_doc_status(s: &str, col: usize) -> rusqlite::Result<DocumentStatus> {
    DocumentStatus::parse(s).ok_or_else(|| sql_enum_error(col, "document status", s))
}

fn sql_enum_error(col: usize, name: &str, value: &str) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(
        col,
        rusqlite::types::Type::Text,
        format!("invalid {name} in sqlite: {value}").into(),
    )
}

async fn import_worker(
    store: Arc<TaskStore>,
    embedder: Arc<dyn Embedder>,
    vector_store: Arc<dyn VectorStore>,
    keyword_index: Arc<dyn KeywordIndex>,
    mut receiver: mpsc::Receiver<ImportJob>,
) {
    while let Some(job) = receiver.recv().await {
        if let Err(error) = process_import(
            store.clone(),
            embedder.clone(),
            vector_store.clone(),
            keyword_index.clone(),
            job,
        )
        .await
        {
            // ImportJob 是 Copy，process_import 只取了副本，job 仍可用。
            let _ = store.fail_import(job, error).await;
        }
    }
}

/// 导入流水线：网络/磁盘 I/O 一律不持 DB 连接，每段只短暂查 SQLite 更新状态。
async fn process_import(
    store: Arc<TaskStore>,
    embedder: Arc<dyn Embedder>,
    vector_store: Arc<dyn VectorStore>,
    keyword_index: Arc<dyn KeywordIndex>,
    job: ImportJob,
) -> Result<(), String> {
    let content = store
        .begin_import(job)
        .await
        .map_err(|error| error.to_string())?;
    let texts = chunk_text(&content);
    if texts.is_empty() {
        return Err("document did not produce any chunks".into());
    }
    store
        .update_import_progress(job.task_id, 25, format!("已切分 {} 个文本块", texts.len()))
        .await
        .map_err(|error| error.to_string())?;

    let references: Vec<&str> = texts.iter().map(String::as_str).collect();
    let vectors = embedder
        .embed(&references)
        .await
        .map_err(|error| error.to_string())?;

    store
        .update_import_progress(
            job.task_id,
            60,
            "向量化完成，正在写入向量库与关键词索引".to_owned(),
        )
        .await
        .map_err(|error| error.to_string())?;

    let records: Vec<ChunkRecord> = texts
        .into_iter()
        .zip(vectors)
        .enumerate()
        .map(|(index, (text, vector))| ChunkRecord {
            document_id: job.document_id,
            chunk_index: index,
            text,
            vector,
        })
        .collect();
    let chunk_count = records.len();
    let bm25_records = records.clone();
    // 向量库与关键词索引相互独立，并发执行 remove + upsert。
    let (vector_remove, keyword_remove) = tokio::join!(
        vector_store.remove_document(job.document_id),
        keyword_index.remove_document(job.document_id),
    );
    vector_remove.map_err(|error| error.to_string())?;
    keyword_remove.map_err(|error| error.to_string())?;
    let (vector_upsert, keyword_upsert) = tokio::join!(
        vector_store.upsert(records),
        keyword_index.upsert(bm25_records),
    );
    vector_upsert.map_err(|error| error.to_string())?;
    keyword_upsert.map_err(|error| error.to_string())?;
    store
        .complete_import(job, chunk_count)
        .await
        .map_err(|error| error.to_string())?;
    // 让出调度，便于事件日志在并发读写下被及时观察。
    yield_now().await;
    Ok(())
}

fn validate_title(title: String) -> Result<String, ApiError> {
    let title = title.trim().to_owned();
    if title.is_empty() {
        return Err(ApiError::bad_request("title cannot be empty"));
    }
    if title.chars().count() > 120 {
        return Err(ApiError::bad_request("title cannot exceed 120 characters"));
    }
    Ok(title)
}

fn validate_document_name(name: String) -> Result<String, ApiError> {
    let name = name.trim().to_owned();
    if name.is_empty() || name.chars().count() > 180 {
        return Err(ApiError::bad_request(
            "document name must contain 1-180 characters",
        ));
    }
    Ok(name)
}

fn page_size(limit: Option<usize>) -> usize {
    limit.unwrap_or(DEFAULT_PAGE_SIZE).clamp(1, MAX_PAGE_SIZE)
}

/// 解析 `documents` 过滤参数：逗号分隔的文档 ID 列表。空字符串/缺省视为不过滤；
/// 任一 token 非合法 u64 即拒绝整个请求，避免静默地少召回。
fn parse_document_filter(raw: Option<&str>) -> Result<Option<HashSet<u64>>, ApiError> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(None);
    }
    let mut ids = HashSet::new();
    for part in raw.split(',') {
        let id = part
            .trim()
            .parse::<u64>()
            .map_err(|_| ApiError::bad_request(format!("invalid document id: {part}")))?;
        ids.insert(id);
    }
    Ok(Some(ids))
}

fn progress_for(status: TaskStatus) -> u8 {
    match status {
        TaskStatus::Todo => 0,
        TaskStatus::InProgress => 50,
        TaskStatus::Done => 100,
    }
}

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before Unix epoch")
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::HashEmbedder;
    use crate::keyword_index::{KeywordIndex, SqliteBm25Index};
    use crate::model::{CreateTaskRequest, EventQuery, Priority, TaskStatus, UpdateTaskRequest};
    use crate::vector_store::{InMemoryVectorStore, VectorStore};

    #[tokio::test]
    async fn stale_updates_are_rejected_and_events_are_recorded() {
        let store = TaskStore::open_in_memory().await.unwrap();
        let created = store
            .create(CreateTaskRequest {
                title: "Learn async Rust".into(),
                priority: Some(Priority::High),
            })
            .await
            .unwrap();
        assert!(
            store
                .update(
                    created.id,
                    UpdateTaskRequest {
                        title: None,
                        status: Some(TaskStatus::Done),
                        priority: None,
                        expected_version: 0
                    }
                )
                .await
                .is_err()
        );
        let updated = store
            .update(
                created.id,
                UpdateTaskRequest {
                    title: None,
                    status: Some(TaskStatus::Done),
                    priority: None,
                    expected_version: created.version,
                },
            )
            .await
            .unwrap();
        assert_eq!(updated.progress, 100);
        assert_eq!(
            store
                .events(EventQuery {
                    after: None,
                    limit: None
                })
                .await
                .unwrap()
                .items
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn indexes_document_and_returns_relevant_chunks() {
        let store = Arc::new(TaskStore::open_in_memory().await.unwrap());
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::new());
        let vector_store: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new());
        let keyword_index: Arc<dyn KeywordIndex> =
            Arc::new(SqliteBm25Index::open_in_memory().await.unwrap());

        let (queued, job) = store
            .queue_document(
                "rust.md".into(),
                "Rust provides ownership and safe concurrency. Tokio powers async services.".into(),
            )
            .await
            .unwrap();
        process_import(
            store.clone(),
            embedder.clone(),
            vector_store.clone(),
            keyword_index.clone(),
            job,
        )
        .await
        .unwrap();

        let query_vector = embedder.embed(&["safe concurrency in Rust"]).await.unwrap();
        let matches = vector_store
            .search(&query_vector[0], 3, &SearchFilter::default())
            .await
            .unwrap();
        let results = store.enrich_search(matches).await.unwrap();

        assert_eq!(
            store.documents().await.unwrap().items[0].status,
            DocumentStatus::Ready
        );
        assert_eq!(queued.document.id, 1);
        assert_eq!(results.len(), 1);
        assert!(results[0].text.contains("ownership"));
        // BM25 索引也应同步写入，关键词检索能命中。
        let keyword_matches = keyword_index
            .search("ownership", 3, &SearchFilter::default())
            .await
            .unwrap();
        assert_eq!(keyword_matches.len(), 1);
        assert_eq!(keyword_matches[0].document_id, 1);
    }

    #[tokio::test]
    async fn events_are_broadcast_to_subscribers() {
        let store = TaskStore::open_in_memory().await.unwrap();
        let mut receiver = store.event_tx.subscribe();
        let created = store
            .create(CreateTaskRequest {
                title: "broadcast me".into(),
                priority: Some(Priority::High),
            })
            .await
            .unwrap();
        let event = receiver.recv().await.unwrap();
        assert_eq!(event.kind, "task.created");
        assert_eq!(event.task_id, created.id);
        assert_eq!(event.task_version, 1);
        // recent_events 应返回包含该事件的历史快照。
        let recent = store.recent_events(10).await.unwrap();
        assert!(recent.iter().any(|e| e.kind == "task.created"));
    }

    #[tokio::test]
    async fn events_ring_buffer_keeps_latest_200() {
        let store = TaskStore::open_in_memory().await.unwrap();
        for i in 0..250 {
            store
                .create(CreateTaskRequest {
                    title: format!("task {i}"),
                    priority: None,
                })
                .await
                .unwrap();
        }
        // 触发器应保留最新 200 条；stats 直查 COUNT(*) 不受分页限制。
        let stats = store.stats().await.unwrap();
        assert_eq!(
            stats.retained_events, 200,
            "events trigger should keep exactly 200"
        );
        // events() API 受 MAX_PAGE_SIZE=100 限制，最多返回 100 条最新事件。
        let page = store
            .events(EventQuery {
                after: None,
                limit: Some(100),
            })
            .await
            .unwrap();
        assert_eq!(page.items.len(), 100);
        // 最旧保留的事件来自第 51 个任务（250-200+1=51），第 1-50 个已被裁剪。
        assert!(page.items[0].task_id >= 50);
    }

    #[tokio::test]
    async fn task_persists_across_reopen() {
        // 模拟重启：写入后重新打开同一文件 DB，数据应在。
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap();
        let (event_tx, _) = broadcast::channel(64);
        let store = TaskStore::open(path, event_tx).await.unwrap();
        store
            .create(CreateTaskRequest {
                title: "persistent task".into(),
                priority: None,
            })
            .await
            .unwrap();
        drop(store);
        let (event_tx, _) = broadcast::channel(64);
        let reopened = TaskStore::open(path, event_tx).await.unwrap();
        let page = reopened
            .list(TaskQuery {
                status: None,
                offset: None,
                limit: None,
            })
            .await
            .unwrap();
        assert_eq!(page.total, 1);
        assert_eq!(page.items[0].title, "persistent task");
    }

    #[test]
    fn document_filter_parsing() {
        assert!(parse_document_filter(None).unwrap().is_none());
        assert!(parse_document_filter(Some("  ")).unwrap().is_none());
        let parsed = parse_document_filter(Some("1, 3, 7")).unwrap().unwrap();
        assert_eq!(parsed, HashSet::from([1, 3, 7]));
        let err = parse_document_filter(Some("1,abc")).unwrap_err();
        assert!(err.to_string().contains("invalid document id"));
    }
}
