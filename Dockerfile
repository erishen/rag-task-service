# syntax=docker/dockerfile:1

# ---------- 构建阶段：编译 release 二进制 ----------
FROM rust:1.91-bookworm AS builder
WORKDIR /app
# lance 系依赖的 prost 构建脚本需要 protoc；.proto well-known types
# 在 libprotobuf-dev 里（/usr/include/google/protobuf/），procprotobuf-compiler 只有二进制
RUN apt-get update \
    && apt-get install -y --no-install-recommends protobuf-compiler libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*
ENV PROTOC=/usr/bin/protoc PROTOC_INCLUDE=/usr/include
# 先只带 Cargo.toml/lock 与一个占位 main，把 lance/prost 等重依赖的
# 编译（含 protoc 生成步骤）固化进这一层缓存，之后改 src 不用重编依赖
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src && printf 'fn main() {}\n' > src/main.rs \
    && cargo build --release --locked && rm -rf src
COPY src ./src
# --locked 与 Cargo.lock 对齐，保证可复现构建
RUN cargo build --release --locked

# ---------- 运行阶段 ----------
FROM debian:bookworm-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/rag-task-service /usr/local/bin/rag-task-service
# 容器内默认绑定 0.0.0.0；数据目录可挂卷持久化
ENV HOST=0.0.0.0 \
    PORT=3000 \
    DATABASE_PATH=/app/data/app.db \
    LANCEDB_PATH=/app/data/lancedb \
    VECTOR_STORE=in_memory
EXPOSE 3000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD ["curl", "-sf", "http://127.0.0.1:3000/health"]
ENTRYPOINT ["rag-task-service"]