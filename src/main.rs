use std::net::SocketAddr;

use rag_task_service::{AppState, Config, router};

#[tokio::main]
async fn main() {
    // 启动时加载 .env 文件（如果存在）
    dotenvy::dotenv().ok();

    let config = Config::from_env();
    let address = format!("{}:{}", config.host, config.port)
        .parse::<SocketAddr>()
        .unwrap_or_else(|_| SocketAddr::from(([127, 0, 0, 1], config.port)));
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .expect("failed to bind TCP listener");

    println!("rag-task-service listening on http://{address}");
    axum::serve(
        listener,
        router(AppState::new(config).await).into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await
    .expect("web server failed unexpectedly");
}
