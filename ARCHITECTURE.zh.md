# 架构文档（RAG Task Service / RAG 检索服务）

> 面向维护者与后继开发者。代码即真相，本文描述**模块职责、关键设计决策与数据流**，
> 不逐行复述代码。配套运行与配置说明见 `README.md` / `README.zh.md`。

## 1. 一句话定位

一个用 Axum 构建的**异步任务服务 + RAG 文档检索服务**：文档上传后切块 → 向量化 →
写入「向量库 + BM25 关键词索引」，检索时两路召回并 RRF 融合，可选 LLM 重排。
任务/文档/事件的元数据统一落在 SQLite；向量与倒排索引按 `VECTOR_STORE` 选择
LanceDB+SQLite（离线默认）或 PostgreSQL+pgvector（规模化）。

## 2. 技术栈

| 类别 | 选型 | 说明 |
|------|------|------|
| Web 框架 | axum 0.8 + tokio | 多层中间件、SSE、multipart |
| HTTP 客户端 | reqwest (rustls) | 调 Ollama embedding / rerank |
| DB | rusqlite + tokio-rusqlite | SQLite 元数据，单写连接串行化 |
| 向量库 | lancedb + arrow | 进程内持久化，可换 pgvector |
| PG 驱动 | sqlx (runtime-tokio + postgres) | `=0.8.6` 锁定，避免版本特性冲突 |
| 中文分词 | jieba-rs | 精确模式 + HMM，词典全局只加载一次 |
| 限流 | governor | per-IP 令牌桶 |
| 错误 | thiserror + 显式枚举 | HTTP 状态码映射锁在类型里 |
| 封送 | async-trait / serde / serde_json | trait 对象化后端 |

## 3. 目录与模块职责

```
src/
├── main.rs          # 入口：加载 .env → Config → AppState → 监听端口
├── lib.rs           # 模块声明，导出 router / Config / AppState
├── config.rs        # 全部配置来自环境变量（VectorStoreKind 三选一）
├── api.rs           # HTTP 路由表 + 中间件装配；SSE 事件流
├── middleware.rs    # 限流（最外层）+ API key 鉴权
├── model.rs         # 领域模型：枚举（状态/模式/优先级）+ DTO
├── error.rs         # BackendError / ApiError：错误 → HTTP 状态码映射
├── store.rs         # TaskStore（SQLite 持久化）+ AppState + import_worker
├── chunk.rs         # 检索数据结构的公共形状：ChunkRecord/Match/SearchFilter
├── rag.rs           # 标题感知切块 + 余弦相似度
├── embedding.rs     # Embedder trait：Hash 伪 embedding / Ollama 真 embedding
├── fusion.rs        # RRF 融合
├── rerank.rs        # Reranker trait：Noop / Llama LLM listwise 重排
├── vector_store.rs  # VectorStore trait：InMemory / LanceDB / pgvector
└── keyword_index.rs # KeywordIndex trait：SQLite BM25 / PostgreSQL BM25
```

依赖方向（无环）：

```
api / middleware
   └─ store.rs (AppState) ── 持有一组 trait 对象 ──┐
                                                    ▼
    AppState ──>> vector_store: Arc<dyn VectorStore>
              ──>> keyword_index: Arc<dyn KeywordIndex>
              ──>> embedder:      Arc<dyn Embedder>
              ──>> reranker:      Option<Arc<dyn Reranker>>
```

后端（向量库/关键词索引/embedder/reranker）全部以 **trait 对象**注入 `AppState`，
切换只在 `AppState::new` 一处发生，其余代码对"用的哪个后端"无感知。
这是本项目内聚的核心机制：**加新后端不改业务代码，只改装配点 + 类型映射**。

## 4. 请求生命周期

### 4.1 任务 CRUD / 事件 / 统计（非 RAG 部分）

- SQLite 单连接，所有 SQL 经 `conn.call(...)` 串行化，等价大锁但数据落盘。
- **乐观并发控制**：`tasks.version` 字段。`PUT`/`DELETE` 要求携带
  `expected_version`，不匹配返回 409，防止并发写静默覆盖（`store.rs:561`）。
- **事件环形缓冲**：`events_trim` 触发器把 `events` 表裁到最近 200 条；
  内存里同发 `broadcast::Sender` 推给 SSE 订阅者。

### 4.2 文档导入（RAG 写入路径）

```
POST /api/documents (multipart)
   → queue_document：SQLite 插 documents + document_contents + 新建导入任务
   → import_tx.send(ImportJob)   // mpsc 队列，单 worker 消费
   → import_worker → process_import：
       1. begin_import        // 状态 Processing，读原文本
       2. chunk_text          // 标题感知切块（800 字符窗口 / 120 重叠）
       3. update_import_progress(25%)
       4. embedder.embed(文本)     // 可批量 64 条调用 Ollama
       5. update_import_progress(60%)
       6. 并发 remove + upsert 到向量库 与 BM25 索引
       7. complete_import(ready)  // 或 fail_import(failed)
```

三大不变式：
- **每段 I/O 不持 SQLite 写连接**：worker 只短暂更新状态，不做长事务。
- **重传幂等**：upsert 前先 `remove_document` 再整文档写入，chunk 集变化不残留。
- **失败可观测**：每个阶段写进度 + 事件，失败落 `error` 字段与 `rag.failed` 事件。

### 4.3 检索（搜索路径）

```
GET /api/search?q=&limit=&mode=&rerank=&documents=
   → 参数校验（q 非空 / documents 逗号分隔 u64 / page_size clamp 1..=100）
   → embed 查询 → query vector
   → SearchFilter{ document_ids } 两路一致下推
   → vector:   vector_store.search(query_vec, pool, filter)
     hybrid:   tokio::join!(vector, keyword)  // 两路并发
               rrf_fuse(...) → 归一化到 [0,1]（k=60 时除 2/61）
   → 可选 rerank：LLM listwise，失败自动回退原顺序
   → enrich_search：进程内 document_names 缓存补名（缺失才查 SQLite）
```

关键点：
- **范围过滤**走 `SearchFilter`，向量侧为 `document_id IN (...)` 下推，BM25 侧为
  SQL 同条件过滤。**两路召回范围必须一致**，否则 RRF 融合串径。
- 检索 latency 敏感、不强锁——文档名缓存用 `RwLock` 短临界区。

## 5. 存储布局

### 5.1 SQLite（任务/文档/事件元数据，默认 `data/app.db`）

```
tasks                # 手动任务 + 导入任务（RAG 导入也产生一条 Import 任务）
documents            # RAG 文档元数据：status/source_chars/chunk_count/error
document_contents    # 原始文本（document_id 外键 CASCADE 删除）
events               # 环形缓冲（触发器裁到 200 条）
chunk_meta  chunk_terms   # BM25 索引（SQLite 后端时与 TaskStore 同一连接）
```

SQLite 连接策略：**单一 `Connection` 复用给 `SqliteBm25Index`**，
避免第二个写连接互相锁死（`from_conn`）。`tokio_rusqlite::Connection` 内部 Arc 可廉价 Clone。

### 5.2 向量库：三选一（`config.rs`）

| `VECTOR_STORE` | 实现 | 表/目录 | 用途 |
|---|---|---|---|
|（缺省/`InMemory`）| `InMemoryVectorStore` | 内存 Vec + 线性扫描 | demo / 测试 |
| `lancedb` | `LanceDbVectorStore` | `data/lancedb/chunks_d{dim}` | 默认离线持久化 |
| `pgvector` | `PgVectorStore` | PG + 扩展 vector，HNSW 索引 | 规模化（数据仍存 PG） |

- **表名按维度区分**（`chunks_d384`）：切换 embedding 模型时维度不匹配的旧表不串写。
- pgvector 模式幂等建扩展/表/HNSW（`CREATE INDEX IF NOT EXISTS ... USING hnsw (vector vector_cosine_ops)`）。
- 相似度统一用**余弦距离**（`<=>`；LanceDB 用 `DistanceType::Cosine`）；返回 score 为 `1 - dist`。

### 5.3 BM25 索引：SQLite / PostgreSQL 两套实现

- **SQLite 版**（`SqliteBm25Index`）：`chunk_meta`（dl）+ `chunk_terms`（postings）两表，
  df 由 `chunk_terms GROUP BY term` 现算（增删后无需维护反链计数）。
  打分在 Rust 内存中完成（与 SQLite key 同 K1=1.2 / B=0.75 / 平滑 idf）。
- **PostgreSQL 版**（`PgBm25Index`）：表结构与 SQLite 对齐，但**打分下推到单条 SQL**：
  `WITH stats(df) → 联合 SUM(LN(1.0+(n-df+0.5)/(df+0.5)) * tf*(K1+1)/(tf+K1*(1-B+B*dl/avgdl))) …`
  只回传 `ORDER BY score DESC LIMIT top-k`。压测下比"全量拉 postings 到应用打分"快 2.9–24 倍，
  支撑数万~数十万 chunk。

两者**打分语义一致**（同 token 序列 + 同 K1/B/平滑 idf），可互相兜底与对拍验证。

## 6. 关键设计决策（为什么）

1. **后端全是 trait**：多后端策略（in-memory/lancedb/pgres）不污染业务，
   切换只在 `AppState::new` 一处，测试用内存实现即得。
2. **错误用类型映射，而非字符串**：`BackendError::{Ollama,VectorStore,KeywordIndex,Rerank}`
   → `ApiError`：上游不可达 503，其余 500。新增后端 = 编译期强制补错误路径。
3. **状态枚举化**：`TaskStatus`/`DocumentStatus`/`Priority` 的每种取值都要在
   `as_str/parse/SQL` 的 `match` 穷尽，写错编译必挂。
4. **Hash 伪 embedding 用 FNV-1a**：`DefaultHasher` 哈希值跨版本不保证，会让落盘的
   旧向量检索错乱；用规范 FNV-1a 保证跨进程/跨版本稳定（`embedding.rs::fnv1a`）。
5. **Hash 向量永不全零**：对仅空白/标点的文本也回退 `vector[0] = 1.0`，
   避免 LanceDB/pgvector 对零向量相似度无定义。
6. **重排是可选的**：网络/解析失败自动降级回原顺序，搜索永不 503（就绪性优先于完美排序）。
7. **SSE 断线续传**：`Last-Event-ID` 优先，缺失则回放最近 20 条；初始化「全量最近」，
   慢消费者丢事件只记日志不断流。

## 7. 部署形态

```
浏览器 ──> frontend (nginx :8080，托管 React SPA，反代 /api /health)
            └─ x-api-key 注头（EventSource 无法自定义）
HTTP 直连 ──> rag-task-service (axum :3000)
              ├─> SQLite + LanceDB（本地/卷 app-data）
              └─> postgres (pgvector/pg17，compose 内网 postgres:5432，宿主 127.0.0.1:5433)
```

- `docker compose up -d` 拉起三个服务；`depends_on: postgres: condition: service_healthy`。
- `DATABASE_URL` 留空则回退 SQLite+LanceDB 默认（本地开发零配置）。
- 生产建议：`API_KEY` 必配、置换默认 `ws/ws` 口令、凭证走密钥管理。

## 8. 可扩展点

- **新增一种向量库**：实现 `VectorStore` trait，`config.rs` 加枚举值，`AppState::new` 加装配。
- **改检索模式**：`SEARCH_MODE` / 请求 `mode` 参数切换 vector/hybrid；RRF 常数 `RRF_K`。
- **加 embedding 模型**：切换维度即新表（自动隔离旧数据）；远程 embedding 由 `EMBEDDING_API_BASE` 打开。
- **加重排器**：实现 `Reranker` trait。

## 9. 测试

- 单元测试内嵌于各模块（`#[cfg(test)]`），覆盖切块边界、RRF 数学、hash 稳定性、
  BM25 中文排序、重排顺序回退、并发更新拒绝、过滤短路等。
- 集成验证：`make verify`（fmt/check/测试/doc/clippy `-D warnings`）。
- PG 真库路径不依赖测试环境，配套 `scripts/bench_scale.py` 做分档压测。

```
make verify    # 完整质量检查（离线 CI 可用）
make docker-build / docker-run / docker-logs / docker-stop
```

## 10. 规模参考

- 压测语料约 17 chunk/篇（8KB md），2 万 doc ≈ 34 万 chunk ≈ 968 万 terms。
- PG hybrid 单次（BM25 下推后）：17 k → 54 ms，34 万 → ~1.5 s；vector 全程 <2.4 ms。
- LanceDB 超过 ~3–5 万 chunk 后向量延迟不再恒定；此规模以上建议 `VECTOR_STORE=pgvector`。
- 压测语料仅有 38 个不同词（病态 hot-term），真实语料分布更丰富会明显更好。