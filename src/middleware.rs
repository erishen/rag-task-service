use std::net::SocketAddr;

use axum::{
    extract::{ConnectInfo, Request, State},
    http::StatusCode,
    middleware::Next,
    response::Response,
};
use governor::{RateLimiter, clock::DefaultClock, state::keyed::DefaultKeyedStateStore};

use crate::store::AppState;

/// per-IP 令牌桶限流器，键为客户端 IP。
pub(crate) type KeyedLimiter =
    RateLimiter<std::net::IpAddr, DefaultKeyedStateStore<std::net::IpAddr>, DefaultClock>;

/// 限流中间件：超出 per-IP 配额返回 429。放最外层，防止暴力请求打爆 Ollama。
pub(crate) async fn rate_limit(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    state
        .rate_limiter
        .check_key(&addr.ip())
        .map_err(|_| StatusCode::TOO_MANY_REQUESTS)?;
    Ok(next.run(request).await)
}

/// API key 中间件：配置了 `API_KEY` 时强制校验 `x-api-key` header；未配置则放行（本地开发）。
pub(crate) async fn api_key_guard(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    if let Some(expected) = state.config.api_key.as_deref()
        && request
            .headers()
            .get("x-api-key")
            .and_then(|value| value.to_str().ok())
            != Some(expected)
    {
        return Err(StatusCode::UNAUTHORIZED);
    }
    Ok(next.run(request).await)
}
