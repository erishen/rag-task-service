use std::env;

use crate::model::SearchMode;

/// 选择哪种持久化向量后端。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorStoreKind {
    InMemory,
    LanceDb,
    /// PostgreSQL + pgvector：适合数万~数十万级文档。需配置 `DATABASE_URL`。
    PgVector,
}

impl VectorStoreKind {
    fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "lancedb" | "lance" => Self::LanceDb,
            "pgvector" | "pg" | "postgres" => Self::PgVector,
            _ => Self::InMemory,
        }
    }
}

/// 进程级配置，全部来自环境变量。
#[derive(Clone, Debug)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub vector_store: VectorStoreKind,
    pub lancedb_path: String,
    pub db_path: String,
    /// PostgreSQL 连接串（`VECTOR_STORE=pgvector` 时必填）。向量后端与关键词索引都使用它。
    pub database_url: Option<String>,
    pub embedding_api_base: Option<String>,
    pub embedding_api_key: Option<String>,
    pub embedding_model: String,
    pub(crate) search_mode: SearchMode,
    pub rrf_k: u32,
    pub rerank_api_base: Option<String>,
    pub rerank_model: Option<String>,
    pub rerank_top_n: usize,
    pub(crate) api_key: Option<String>,
    pub(crate) rate_limit_per_sec: u32,
}

impl Config {
    pub fn from_env() -> Self {
        let port = env::var("PORT")
            .ok()
            .and_then(|value| value.parse::<u16>().ok())
            .unwrap_or(3000);
        let host = env::var("HOST").unwrap_or_else(|_| "127.0.0.1".to_owned());
        let vector_store = VectorStoreKind::parse(&env::var("VECTOR_STORE").unwrap_or_default());
        let lancedb_path = env::var("LANCEDB_PATH").unwrap_or_else(|_| "data/lancedb".to_owned());
        let db_path = env::var("DATABASE_PATH").unwrap_or_else(|_| "data/app.db".to_owned());
        let database_url = env::var("DATABASE_URL")
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty());
        let embedding_api_base = env::var("EMBEDDING_API_BASE")
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty());
        let embedding_api_key = env::var("EMBEDDING_API_KEY")
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty());
        let embedding_model =
            env::var("EMBEDDING_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_owned());

        let search_mode = match env::var("SEARCH_MODE")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "vector" => SearchMode::Vector,
            _ => SearchMode::Hybrid,
        };
        let rrf_k = env::var("RRF_K")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(60);
        // 重排端点缺省回退到 embedding 端点（本地 Ollama 同址）。
        let rerank_api_base = env::var("RERANK_API_BASE")
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
            .or_else(|| embedding_api_base.clone());
        let rerank_model = env::var("RERANK_MODEL")
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty());
        let rerank_top_n = env::var("RERANK_TOP_N")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(20);
        let api_key = env::var("API_KEY")
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty());
        let rate_limit_per_sec = env::var("RATE_LIMIT_PER_SEC")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(10);

        Self {
            host,
            port,
            vector_store,
            lancedb_path,
            db_path,
            database_url,
            embedding_api_base,
            embedding_api_key,
            embedding_model,
            search_mode,
            rrf_k,
            rerank_api_base,
            rerank_model,
            rerank_top_n,
            api_key,
            rate_limit_per_sec,
        }
    }

    /// 配置了 embedding API 端点时使用真实语义 embedding，否则退回本地 hash embedding。
    pub(crate) fn use_remote_embedding(&self) -> bool {
        self.embedding_api_base.is_some()
    }

    /// 配置了 `RERANK_MODEL`（且端点可用，端点可回退到 embedding 端点）时启用 LLM 重排。
    pub(crate) fn use_rerank(&self) -> bool {
        self.rerank_model.is_some() && self.rerank_api_base.is_some()
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_owned(),
            port: 3000,
            vector_store: VectorStoreKind::InMemory,
            lancedb_path: "data/lancedb".to_owned(),
            db_path: "data/app.db".to_owned(),
            database_url: None,
            embedding_api_base: None,
            embedding_api_key: None,
            embedding_model: "nomic-embed-text".to_owned(),
            search_mode: SearchMode::Hybrid,
            rrf_k: 60,
            rerank_api_base: None,
            rerank_model: None,
            rerank_top_n: 20,
            api_key: None,
            rate_limit_per_sec: 10,
        }
    }
}
