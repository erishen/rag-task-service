use std::convert::Infallible;

use async_stream::stream;
use axum::{
    Json, Router,
    extract::{Multipart, Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::sse::{Event, KeepAlive, Sse},
    routing::{delete, get},
};
use futures::stream::Stream;
use serde::Serialize;
use tokio::sync::broadcast::error::RecvError;

use crate::{
    AppState,
    error::ApiError,
    model::{
        CreateTaskRequest, DeleteTaskQuery, DocumentImportResponse, DocumentPage, EventPage,
        EventQuery, SearchQuery, SearchResponse, Task, TaskPage, TaskQuery, TaskStats,
        UpdateTaskRequest,
    },
};

const MAX_DOCUMENT_BYTES: usize = 5 * 1024 * 1024;

#[derive(Serialize)]
struct ServiceInfo {
    name: &'static str,
    message: &'static str,
    endpoints: &'static [&'static str],
}

#[derive(Serialize)]
struct HealthStatus {
    status: &'static str,
    task_count: usize,
}

pub fn router(state: AppState) -> Router {
    // /api/* 路由组：先限流（最外层）再鉴权，保护 Ollama 不被暴力请求打爆。
    let api = Router::<AppState>::new()
        .route("/tasks", get(list_tasks).post(create_task))
        .route(
            "/tasks/{id}",
            get(get_task)
                .put(update_task)
                .patch(update_task)
                .delete(delete_task),
        )
        .route("/events", get(list_events))
        .route("/events/stream", get(stream_events))
        .route("/stats", get(stats))
        .route(
            "/documents",
            get(list_documents)
                .post(upload_document)
                .delete(reset_documents),
        )
        .route("/documents/{id}", delete(delete_document))
        .route("/search", get(search))
        .route_layer(axum::middleware::from_fn_with_state(
            state.clone(),
            crate::middleware::api_key_guard,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            state.clone(),
            crate::middleware::rate_limit,
        ));

    Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .nest("/api", api)
        .with_state(state)
}

async fn index() -> Json<ServiceInfo> {
    Json(ServiceInfo {
        name: "rag-task-service",
        message: "An async task API with local RAG document indexing.",
        endpoints: &[
            "GET/POST /api/tasks",
            "GET/PATCH/DELETE /api/tasks/{id}",
            "POST/GET /api/documents",
            "GET /api/search?q=...",
            "GET /api/events",
            "GET /api/stats",
        ],
    })
}

async fn health(State(state): State<AppState>) -> Result<Json<HealthStatus>, ApiError> {
    Ok(Json(HealthStatus {
        status: "ok",
        task_count: state.store.task_count().await?,
    }))
}

async fn create_task(
    State(state): State<AppState>,
    Json(request): Json<CreateTaskRequest>,
) -> Result<(StatusCode, Json<Task>), ApiError> {
    Ok((
        StatusCode::CREATED,
        Json(state.store.create(request).await?),
    ))
}

async fn list_tasks(
    State(state): State<AppState>,
    Query(query): Query<TaskQuery>,
) -> Result<Json<TaskPage>, ApiError> {
    Ok(Json(state.store.list(query).await?))
}

async fn get_task(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> Result<Json<Task>, ApiError> {
    Ok(Json(state.store.get(id).await?))
}

async fn update_task(
    State(state): State<AppState>,
    Path(id): Path<u64>,
    Json(request): Json<UpdateTaskRequest>,
) -> Result<Json<Task>, ApiError> {
    Ok(Json(state.store.update(id, request).await?))
}

async fn delete_task(
    State(state): State<AppState>,
    Path(id): Path<u64>,
    Query(query): Query<DeleteTaskQuery>,
) -> Result<StatusCode, ApiError> {
    state.store.delete(id, query.expected_version).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn list_events(
    State(state): State<AppState>,
    Query(query): Query<EventQuery>,
) -> Result<Json<EventPage>, ApiError> {
    Ok(Json(state.store.events(query).await?))
}

/// SSE 事件流：连接建立时按 `Last-Event-ID` 回放缺失事件（无该 header 则回放最近 20 条），
/// 之后实时推送新增事件。浏览器 EventSource 断线重连会自动带上 `Last-Event-ID`，实现续传不丢。
/// 鉴权：浏览器原生 EventSource 无法自定义 Header，依赖反向代理注入 `x-api-key`（vite 已配置）。
async fn stream_events(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let snapshot = match headers
        .get("last-event-id")
        .and_then(|value| value.to_str().ok())
    {
        Some(last_id) => match last_id.trim().parse::<u64>() {
            Ok(id) => state.store.events_after(id).await?,
            Err(_) => state.store.recent_events(20).await?,
        },
        None => state.store.recent_events(20).await?,
    };
    let mut receiver = state.event_tx.subscribe();
    let events = stream! {
        for event in snapshot {
            match Event::default().id(event.id.to_string()).json_data(&event) {
                Ok(serialized) => yield Ok(serialized),
                Err(error) => eprintln!("sse serialize event failed: {error}"),
            }
        }
        loop {
            match receiver.recv().await {
                Ok(event) => {
                    match Event::default().id(event.id.to_string()).json_data(&event) {
                        Ok(serialized) => yield Ok(serialized),
                        Err(error) => eprintln!("sse serialize event failed: {error}"),
                    }
                }
                // 慢消费者丢了事件：日志提示前端可重新拉全量快照，SSE 本身保持连接。
                Err(RecvError::Lagged(skipped)) => {
                    eprintln!("sse consumer lagged, skipped {skipped} events");
                }
                // 所有 Sender 已 drop（服务关闭），结束流。
                Err(RecvError::Closed) => break,
            }
        }
    };
    Ok(Sse::new(events).keep_alive(KeepAlive::default()))
}

async fn stats(State(state): State<AppState>) -> Result<Json<TaskStats>, ApiError> {
    Ok(Json(state.store.stats().await?))
}

async fn list_documents(State(state): State<AppState>) -> Result<Json<DocumentPage>, ApiError> {
    Ok(Json(state.store.documents().await?))
}

async fn upload_document(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<(StatusCode, Json<DocumentImportResponse>), ApiError> {
    let mut upload = None;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| ApiError::bad_request(error.to_string()))?
    {
        if field.name() == Some("file") {
            let name = field.file_name().unwrap_or("upload.txt").to_owned();
            let bytes = field
                .bytes()
                .await
                .map_err(|error| ApiError::bad_request(error.to_string()))?;
            if bytes.len() > MAX_DOCUMENT_BYTES {
                return Err(ApiError::bad_request("document exceeds 5 MiB limit"));
            }
            upload = Some((name, bytes));
            break;
        }
    }
    let (name, bytes) =
        upload.ok_or_else(|| ApiError::bad_request("multipart field 'file' is required"))?;
    if !matches!(name.rsplit('.').next(), Some("txt" | "md" | "markdown")) {
        return Err(ApiError::bad_request(
            "only UTF-8 .txt, .md, and .markdown files are supported",
        ));
    }
    let content = String::from_utf8(bytes.to_vec())
        .map_err(|_| ApiError::bad_request("document must be UTF-8 text"))?;
    let (response, job) = state.store.queue_document(name, content).await?;
    state.enqueue_import(job).await?;
    Ok((StatusCode::ACCEPTED, Json(response)))
}

async fn reset_documents(State(state): State<AppState>) -> Result<StatusCode, ApiError> {
    state.reset().await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn delete_document(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> Result<StatusCode, ApiError> {
    state.delete_document(id).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn search(
    State(state): State<AppState>,
    Query(query): Query<SearchQuery>,
) -> Result<Json<SearchResponse>, ApiError> {
    Ok(Json(state.search(query).await?))
}
