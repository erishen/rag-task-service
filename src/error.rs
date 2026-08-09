use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;

#[derive(Serialize)]
struct ErrorResponse {
    error: &'static str,
    message: String,
}

/// 后端（embedding / 向量库 / 重排）调用失败。Ollama 与重排上游不可达走 503，其余走 500。
#[derive(Debug, thiserror::Error)]
pub(crate) enum BackendError {
    #[error("ollama unavailable: {0}")]
    Ollama(String),
    #[error("vector store failed: {0}")]
    VectorStore(String),
    #[error("keyword index failed: {0}")]
    KeywordIndex(String),
    #[error("rerank failed: {0}")]
    Rerank(String),
}

impl BackendError {
    pub(crate) fn ollama(message: impl Into<String>) -> Self {
        Self::Ollama(message.into())
    }

    pub(crate) fn vector_store(message: impl Into<String>) -> Self {
        Self::VectorStore(message.into())
    }

    pub(crate) fn keyword_index(message: impl Into<String>) -> Self {
        Self::KeywordIndex(message.into())
    }

    pub(crate) fn rerank(message: impl Into<String>) -> Self {
        Self::Rerank(message.into())
    }
}

#[derive(Debug)]
pub(crate) enum ApiError {
    BadRequest(String),
    NotFound(String),
    Conflict(String),
    Internal(String),
    ServiceUnavailable(String),
}

impl ApiError {
    pub(crate) fn bad_request(message: impl Into<String>) -> Self {
        Self::BadRequest(message.into())
    }

    pub(crate) fn not_found(message: impl Into<String>) -> Self {
        Self::NotFound(message.into())
    }

    pub(crate) fn conflict(message: impl Into<String>) -> Self {
        Self::Conflict(message.into())
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadRequest(message) => write!(f, "bad request: {message}"),
            Self::NotFound(message) => write!(f, "not found: {message}"),
            Self::Conflict(message) => write!(f, "version conflict: {message}"),
            Self::Internal(message) => write!(f, "internal error: {message}"),
            Self::ServiceUnavailable(message) => write!(f, "service unavailable: {message}"),
        }
    }
}

impl From<BackendError> for ApiError {
    fn from(error: BackendError) -> Self {
        match error {
            BackendError::Ollama(message) | BackendError::Rerank(message) => {
                Self::ServiceUnavailable(message)
            }
            other => Self::internal(other.to_string()),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error, message) = match self {
            Self::BadRequest(message) => (StatusCode::BAD_REQUEST, "bad_request", message),
            Self::NotFound(message) => (StatusCode::NOT_FOUND, "not_found", message),
            Self::Conflict(message) => (StatusCode::CONFLICT, "version_conflict", message),
            Self::Internal(message) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", message)
            }
            Self::ServiceUnavailable(message) => (
                StatusCode::SERVICE_UNAVAILABLE,
                "service_unavailable",
                message,
            ),
        };
        (status, Json(ErrorResponse { error, message })).into_response()
    }
}
