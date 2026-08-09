use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TaskStatus {
    Todo,
    InProgress,
    Done,
}

impl TaskStatus {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Todo => "todo",
            Self::InProgress => "in_progress",
            Self::Done => "done",
        }
    }

    pub(crate) fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "todo" => Self::Todo,
            "in_progress" => Self::InProgress,
            "done" => Self::Done,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TaskKind {
    Manual,
    DocumentImport,
}

impl TaskKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Manual => "manual",
            Self::DocumentImport => "document_import",
        }
    }

    pub(crate) fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "manual" => Self::Manual,
            "document_import" => Self::DocumentImport,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Priority {
    Low,
    #[default]
    Normal,
    High,
}

impl Priority {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Normal => "normal",
            Self::High => "high",
        }
    }

    pub(crate) fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "low" => Self::Low,
            "normal" => Self::Normal,
            "high" => Self::High,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Task {
    pub(crate) id: u64,
    pub(crate) title: String,
    pub(crate) status: TaskStatus,
    pub(crate) kind: TaskKind,
    pub(crate) priority: Priority,
    pub(crate) progress: u8,
    pub(crate) detail: Option<String>,
    pub(crate) version: u64,
    pub(crate) created_at: u64,
    pub(crate) updated_at: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct CreateTaskRequest {
    pub(crate) title: String,
    pub(crate) priority: Option<Priority>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct UpdateTaskRequest {
    pub(crate) title: Option<String>,
    pub(crate) status: Option<TaskStatus>,
    pub(crate) priority: Option<Priority>,
    pub(crate) expected_version: u64,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TaskQuery {
    pub(crate) status: Option<TaskStatus>,
    pub(crate) offset: Option<usize>,
    pub(crate) limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct DeleteTaskQuery {
    pub(crate) expected_version: u64,
}

#[derive(Debug, Deserialize)]
pub(crate) struct EventQuery {
    pub(crate) after: Option<u64>,
    pub(crate) limit: Option<usize>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SearchMode {
    Vector,
    #[default]
    Hybrid,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SearchQuery {
    pub(crate) q: String,
    pub(crate) limit: Option<usize>,
    /// 检索模式：`vector` 纯向量，`hybrid` BM25+向量 RRF 融合。缺省按 `Config` 默认（hybrid）。
    pub(crate) mode: Option<SearchMode>,
    /// 是否对召回结果做 LLM 重排。缺省按 `Config::use_rerank()`。
    pub(crate) rerank: Option<bool>,
    /// 检索范围过滤：逗号分隔的文档 ID 列表。缺省/空串表示全库检索。
    pub(crate) documents: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct TaskPage {
    pub(crate) items: Vec<Task>,
    pub(crate) total: usize,
    pub(crate) next_offset: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct TaskEvent {
    pub(crate) id: u64,
    pub(crate) kind: String,
    pub(crate) task_id: u64,
    pub(crate) task_version: u64,
    pub(crate) occurred_at: u64,
}

#[derive(Serialize)]
pub(crate) struct EventPage {
    pub(crate) items: Vec<TaskEvent>,
    pub(crate) latest_event_id: u64,
}

#[derive(Serialize)]
pub(crate) struct TaskStats {
    pub(crate) total: usize,
    pub(crate) todo: usize,
    pub(crate) in_progress: usize,
    pub(crate) done: usize,
    pub(crate) retained_events: usize,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DocumentStatus {
    Queued,
    Processing,
    Ready,
    Failed,
}

impl DocumentStatus {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Processing => "processing",
            Self::Ready => "ready",
            Self::Failed => "failed",
        }
    }

    pub(crate) fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "queued" => Self::Queued,
            "processing" => Self::Processing,
            "ready" => Self::Ready,
            "failed" => Self::Failed,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RagDocument {
    pub(crate) id: u64,
    pub(crate) name: String,
    pub(crate) status: DocumentStatus,
    pub(crate) source_chars: usize,
    pub(crate) chunk_count: usize,
    pub(crate) error: Option<String>,
    pub(crate) created_at: u64,
    pub(crate) updated_at: u64,
}

#[derive(Serialize)]
pub(crate) struct DocumentImportResponse {
    pub(crate) document: RagDocument,
    pub(crate) task: Task,
}

#[derive(Serialize)]
pub(crate) struct DocumentPage {
    pub(crate) items: Vec<RagDocument>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SearchResult {
    pub(crate) document_id: u64,
    pub(crate) document_name: String,
    pub(crate) chunk_index: usize,
    pub(crate) text: String,
    pub(crate) score: f32,
}

#[derive(Serialize)]
pub(crate) struct SearchResponse {
    pub(crate) query: String,
    pub(crate) items: Vec<SearchResult>,
}
