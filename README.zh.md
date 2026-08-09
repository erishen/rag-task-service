# RAG Task Service

一个使用 [Axum](https://github.com/tokio/axum) 构建的异步任务服务与 RAG 文档检索流水线。任务、文档元数据与事件持久化到 SQLite（WAL 模式 + 事件环形缓冲），通过任务版本号实现乐观并发控制，避免并发写入时的静默覆盖。内置 RAG 流水线：上传文档 → 切块 → 向量化 → 写入向量库 → 检索。检索支持纯向量召回、BM25+向量 RRF 混合召回，以及可选的 LLM 重排。embedding、向量库与重排均可通过环境变量在本地默认实现与持久化后端之间切换；BM25 关键词索引与向量库共用同一个后端——可选 **SQLite + LanceDB**（离线默认）或 **PostgreSQL + pgvector**（`VECTOR_STORE=pgvector`），重启后无需重建即可直接检索。

## 为什么选 Rust

- **错误是类型化的。** `BackendError`（`src/error.rs`）把上游失败（`Ollama`、`Rerank`）与内部后端失败（`VectorStore`、`KeywordIndex`）区分开，HTTP 状态码的映射被锁进 `impl From<BackendError> for ApiError`：上游不可达 → 503，其余 → 500。运行时不可能"随便挑一个"错误码；新增一个后端，就不得不在编译期显式补齐它的错误映射路径。
- **不存在非法状态。** `DocumentStatus`、`TaskStatus`、`Priority` 都是枚举（`src/model.rs`）：手滑写个 `"ready!"` 根本编译不过。对它们的每个 `match` 都是穷尽的——新增一个枚举值，编译器会强制你逐处补全（`as_str`、`parse`、SQL 列）处理。
- **歧义由类型表达。** 可选入参是 `Option<T>`（`limit: Option<usize>`、`mode: Option<SearchMode>`），"没传"和"传 0"是两种类型，调用方无法混淆。
- **工具链把底线钉死。** `cargo clippy -- -D warnings`、`cargo fmt --check` 与单测套件统一由 `make verify` 执行；warning 即编译错误，错误处理上的收益贯穿整条流水线。

## 启动

```bash
make run
```

服务默认监听 `http://127.0.0.1:3000`，也可通过 `PORT` 指定端口：

```bash
make run PORT=8080
```

### Docker 启动（双容器：nginx 前端 + 后端）

```bash
make docker-build   # 构建两个镜像（首次构建需几分钟）
make docker-run     # 构建并后台启动
make docker-logs    # 跟踪容器日志
make docker-stop    # 停止并移除容器（数据保留）
```

- 前端（nginx 托管构建好的 React 应用）：`http://127.0.0.1:8080`
- 后端 API（直连）：`http://127.0.0.1:3000`

前端容器在 compose 网络内把 `/api` 与 `/health` 反代到后端，并从 `.env` 注入 `x-api-key`（浏览器 EventSource 无法自定义 Header），因此 SSE 在容器里无需额外处理。compose 还内置一个 `postgres` 容器（pgvector/pg17，宿主机 `127.0.0.1:5433`）；`VECTOR_STORE=pgvector` 时后端通过网络内的 `postgres://ws:ws@postgres:5432/ws` 连接它。SQLite 与 LanceDB 数据持久化到命名卷 `app-data`，PostgreSQL 数据落 `pg-data` 卷；`.env` 配置（`API_KEY`、`VECTOR_STORE` 等）自动注入两个容器。

查看完整的开发、测试、检查与发布命令：

```bash
make help
```

## 环境变量配置

复制示例配置并编辑：

```bash
cp .env.example .env
```

### 生成 API Key

```bash
# 方式一：openssl（推荐）
openssl rand -hex 32

# 方式二：Python
python3 -c "import secrets; print(secrets.token_hex(32))"

# 方式三：Node.js
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

将生成的密钥填入 `.env`：

```bash
API_KEY=your-generated-key-here
```

> **注意**：`.env` 已加入 `.gitignore`，不会被提交。生产环境请妥善保管 API Key。

## 配置

全部通过环境变量控制，未设置时使用离线默认值（hash 伪 embedding + 内存向量库），无需任何外部依赖即可运行：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PORT` | `3000` | 服务监听端口 |
| `DATABASE_PATH` | `data/app.db` | SQLite 数据库路径；任务、文档元数据与事件持久化于此 |
| `VECTOR_STORE` | `in_memory` | 向量库后端：`in_memory`、`lancedb`（本地 Lance 文件）或 `pgvector`（PostgreSQL，需配 `DATABASE_URL`） |
| `LANCEDB_PATH` | `data/lancedb` | LanceDB 数据目录（仅 `VECTOR_STORE=lancedb` 时生效） |
| `DATABASE_URL` | _空_ | PostgreSQL 连接串（`VECTOR_STORE=pgvector` 时必填）；向量库与关键词索引共用它 |
| `EMBEDDING_API_BASE` | _空_ | embedding API 端点；设置后切换到真实语义 embedding，留空则用本地 hash 伪 embedding（FNV-1a bigram 哈希，跨进程稳定） |
| `EMBEDDING_API_KEY` | _空_ | embedding API 鉴权 token（本地 Ollama 无需） |
| `EMBEDDING_MODEL` | `nomic-embed-text` | embedding 模型名 |
| `SEARCH_MODE` | `hybrid` | 检索模式：`hybrid`（BM25+向量已融合）或 `vector`（纯向量） |
| `RRF_K` | `60` | RRF 融合常数，越大越平滑双路排名差异 |
| `RERANK_MODEL` | _空_ | 重排模型名（如 `llama3.1:8b`）；设置后启用 LLM 重排 |
| `RERANK_API_BASE` | _空_ | 重排 API 端点；缺省回退到 `EMBEDDING_API_BASE` |
| `RERANK_TOP_N` | `5` | 送入重排的候选数量 |

### 接入本地 Ollama 做真实语义 embedding

启动 Ollama 并拉取一个 embedding 模型，然后指定端点即可：

```bash
ollama pull nomic-embed-text
ollama serve
EMBEDDING_API_BASE=http://127.0.0.1:11434/v1 EMBEDDING_MODEL=nomic-embed-text \
  VECTOR_STORE=lancedb make run
```

Ollama 的 `/v1/embeddings` 端点兼容 OpenAI 协议，因此把 `EMBEDDING_API_BASE` 指向任意兼容服务同样可用。`OllamaEmbedder` 在构造时会发送一个探针请求确定输出维度，并按维度命名 LanceDB 表（`chunks_d{dim}`），因此切换模型不会静默写入错误维度的向量。

### 持久化向量库（LanceDB）

`VECTOR_STORE=lancedb` 时，向量写入 `LANCEDB_PATH` 指定的目录，进程重启后仍可检索；任务、文档元数据与事件持久化到 `DATABASE_PATH` 指定的 SQLite 文件，进程重启后完整恢复。BM25 关键词索引（postings 与文档长度统计）同样落 SQLite（`chunk_meta` / `chunk_terms` 表），启动不再需要全量重建，混合搜索在重启后立即可用。

> 构建提示：本地构建需要 Rust ≥ 1.91；LanceDB 间接依赖 `lzma-sys`，精简环境若报错请先安装 `xz`（macOS：`brew install xz`）与 `protoc`（lance 的 prost 构建脚本需要，macOS：`brew install protobuf`）。Dockerfile 已代为安装 `protobuf-compiler` + `libprotobuf-dev`（well-known `.proto` 在此包里），可直接 `make docker-build`。

### 规模化：PostgreSQL + pgvector

`VECTOR_STORE=pgvector` 把向量库与 BM25 关键词索引一起迁进同一个带 `vector` 扩展的 PostgreSQL 库（`chunk_meta` / `chunk_terms` / `chunks_d{dim}` 表），是数万~数十万级文档的目标后端——此时进程内文件型存储的检索延迟开始退化。服务启动时幂等地创建扩展、建表，并为 chunk 表建 HNSW 索引（`vector_cosine_ops`）。

```bash
docker compose up -d postgres            # compose 已内置 pgvector/postgres 服务
VECTOR_STORE=pgvector \
  DATABASE_URL=postgres://ws:ws@127.0.0.1:5433/ws \
  make run
```

> 在 compose 内运行时把 `DATABASE_URL` 指向服务名：`postgres://ws:ws@postgres:5432/ws`（自动从 `.env` 读取）。相似度算子可选点积/余弦/L2，当前实现用余弦距离（`<=>`）。

### 规模化下的性能

BM25 打分在 **PostgreSQL 内完成**：一条 SQL（`WITH` stats/df 两段	CTE 的 JOIN）逐 chunk 计算完整 BM25 分数（`k1 = 1.2`、`b = 0.75`）并直接 `LIMIT` 取 Top-N，驱动层不再把 postings 拉到应用端。向量检索走 HNSW 索引，延迟恒定。release 构建在合成语料（约 8KB/篇 → 17 chunk/篇，2 万篇 ≈ 34 万 chunk）上的实测：

| chunk 规模 | hybrid（BM25 下推后） | vector（HNSW） |
| ---------- | --------------------- | -------------- |
| 1.7 万      | 54 ms                 | <2.4 ms        |
| 5.1 万      | 134 ms                | <2.4 ms        |
| 10.2 万     | 426 ms                | <2.4 ms        |
| 17 万       | 580 ms                | <2.4 ms        |
| 25.5 万     | 939 ms                | <2.4 ms        |
| 34 万       | 1.5 s                 | <2.4 ms        |

SQL 下推之前，每文档 postings 会被拉到应用端用 Rust 重新打分——17 万 chunk 时耗时 1306 ms、34 万 chunk 时 4.4 s；下推后整体快 **2.9–24 倍**。同一语料在 LanceDB 上超过约 3–5 万 chunk 后开始退化（5.1 万 chunk：vector 130 ms、hybrid 617 ms）。经验法则：数万 chunk 以内 LanceDB 足够；超过就切换到 `VECTOR_STORE=pgvector`。

> 注意：压测语料只有 38 个不同词，每个词会命中约 32 万 chunk——这是病态的过热词场景，真实语料分布更丰富会好得多；任意规模下做单文档过滤后 hybrid 仍能回到 31–53 ms。打分一致性：与文件型/SQLite 索引相比，PG hybrid 在过滤集内返回顺序一致（全库时仅 RRF 并列次序不同）。

### 切块与文档元数据缓存

文档按 Markdown `#` 标题分节，节内再按 800 字符窗口切块（重叠 120）；无标题的纯文本回落为整篇窗口切块。每个块会把标题祖先链（如 `## LanceDB`）作为前缀拼入，语义检索与 BM25 都能直接命中「标题即答案」型查询。检索结果补全文档名时优先命中进程内缓存，只有缓存缺失的 ID 才回查 SQLite，避免每次搜索都查一遍文档表。

## 检索：混合搜索与重排

`/api/search` 支持两种召回模式（由 `SEARCH_MODE` 或请求参数 `mode` 控制）：

- **`vector`**：纯向量召回，用查询向量在配置的向量库（LanceDB 或 pgvector）中做最近邻搜索。
- **`hybrid`**（默认）：BM25 关键词检索（`jieba-rs` 中文分词）与向量检索**并发**执行，再用 RRF（Reciprocal Rank Fusion，常数 `RRF_K`）融合两路排名。RRF 原始分归一化到 `[0,1]`，保持前端百分比语义。混合模式对中文短查询尤其有效——弥补了 embedding 在词面匹配上的盲区。

支持 **metadata 文档范围过滤**：参数 `documents` 传逗号分隔的文档 ID，检索（含 hybrid 两路）只召回这些文档的 chunk。过滤谓词在向量检索侧下推给 LanceDB 或 pgvector（`document_id IN (...)`），BM25 侧同步跳过范围外文档，保证 RRF 两路召回范围一致。非法 ID 返回 `400`。

可选启用 **LLM 重排**（配置 `RERANK_MODEL` 后生效）：取融合后前 `RERANK_TOP_N` 个候选用 LLM 做 listwise 重排，再按相关性截断。重排是**可选增强**：LLM 调用失败（超时/网络/输出不可解析）时自动降级回原顺序，搜索始终返回结果而非报错。

> 性能参考（本地 Ollama，debug 构建）：纯向量/混合单次检索 ~90ms；开启 `llama3.1:8b` 重排单次约 5–11s。重排并发受 Ollama 单流处理限制，生产环境建议调大 `OLLAMA_NUM_PARALLEL` 或使用独立重排服务。

## 验证服务

运行完整质量检查（离线，使用默认 hash embedding + 内存向量库）：

```bash
make verify
```

运行 API 冒烟测试（任务创建、乐观锁更新、事件记录、删除）：

```bash
make api-smoke
```

运行 RAG 冒烟测试（上传文档、索引、检索；默认走 LanceDB + hash embedding，配置 `EMBEDDING_API_BASE` 后自动走 Ollama）：

```bash
make rag-smoke
```

运行检索评估（`fixtures/eval.json` 内置带期望文档标注的查询集；起临时环境、上传样例文档后对比 vector / hybrid 命中率，并验证 metadata 过滤不越界。任一模式 recall@k 低于阈值（缺省 1.0）或过滤越界时以非零退出，可作回归闸门）：

```bash
make eval
EVAL_MIN_RECALL=0.7 make eval   # 放宽阈值
```

## React 前端

前端位于 `frontend/`，通过 Vite 开发服务器代理 `/api` 和 `/health` 到 Axum 服务。首次使用时安装依赖：

```bash
make frontend-install
```

之后使用一个命令同时启动后端与前端；按下 `Ctrl+C` 会一并停止：

```bash
make dev
```

随后访问 `http://127.0.0.1:5173`。页面包含四部分：

- **任务控制台**：看板、统计卡片与事件时间线。右上角“添加测试任务”会创建三条样例任务并推进状态。
- **RAG 文档检索**：上传 `.txt`/`.md`/`.markdown` 文档，观察导入任务进度推进与文档状态切换（排队中 → 处理中 → 已就绪），随后在右侧做语义检索并查看带相似度得分的结果。
- **事件实时流**：通过 SSE（`/api/events/stream`）实时展示任务与文档导入事件的推送，导入类事件会即时刷新文档列表（SSE 断开时自动降级为 2 秒轮询）。
- **工作原理**：端到端链路说明。

生产构建可运行：

```bash
make frontend-build
```

## 接口

- `GET /`：服务信息
- `GET /health`：健康检查与任务数量
- `GET /api/tasks`：任务分页列表，支持 `status`、`offset`、`limit` 参数
- `POST /api/tasks`：创建任务
- `GET /api/tasks/{id}`：查询单个任务
- `PATCH /api/tasks/{id}`：更新任务；必须携带当前 `expected_version`
- `DELETE /api/tasks/{id}?expected_version=1`：删除任务
- `GET /api/events`：读取保留的任务变更事件（最多保留 200 条）。支持 `after`（增量起点事件 id）与 `limit` 参数
- `GET /api/events/stream`：SSE 事件流。连接时按请求头 `Last-Event-ID` 回放该事件之后的所有事件（无该请求头回放最近 20 条），之后实时推送任务与文档导入事件（`data` 为事件 JSON，`id` 为事件 id，KeepAlive 心跳约 15 秒）。浏览器 EventSource 断线重连会自动携带 `Last-Event-ID` 实现续传。鉴权需由反向代理注入 `x-api-key`（浏览器 EventSource 无法自定义 Header；vite 开发代理已内置，Docker 下 nginx 前端容器会自动注入 `.env` 中的 key）
- `GET /api/stats`：任务状态统计
- `POST /api/documents`：multipart 上传文本文档（`.txt`/`.md`/`.markdown`，≤ 5 MiB），返回文档与导入任务
- `GET /api/documents`：文档列表与状态
- `GET /api/search?q=...&mode=hybrid&rerank=true&limit=5&documents=1,3`：检索，返回命中的文本块与相似度得分。`mode` 可选 `hybrid`（默认，BM25+向量 RRF）或 `vector`；`rerank=true` 启用 LLM 重排（需配置 `RERANK_MODEL`）；`limit` 控制返回数量；`documents` 把召回范围圈定到指定文档 ID

## 示例

```bash
# 创建任务
curl -X POST http://127.0.0.1:3000/api/tasks \
  -H 'content-type: application/json' \
  -d '{"title":"实现并发任务服务","priority":"high"}'

# 以版本 1 更新任务。并发客户端如果持有旧版本，会收到 409 Conflict。
curl -X PATCH http://127.0.0.1:3000/api/tasks/1 \
  -H 'content-type: application/json' \
  -d '{"status":"in_progress","expected_version":1}'

# 上传文档并检索
curl -X POST http://127.0.0.1:3000/api/documents -F "file=@notes.md"
curl http://127.0.0.1:3000/api/documents
curl "http://127.0.0.1:3000/api/search?q=safe+concurrency"

# 查看事件与统计
curl http://127.0.0.1:3000/api/events
curl http://127.0.0.1:3000/api/stats
```

<div align="center">[🇺🇸 README (English)](README.md)</div>