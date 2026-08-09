# RAG Task Service

An asynchronous task service and RAG document retrieval pipeline built with [Axum](https://github.com/tokio/axum). Tasks, document metadata, and events persist to SQLite (WAL mode + event ring buffer); optimistic concurrency via task version numbers prevents silent overwrites between concurrent writers. A built-in RAG pipeline handles upload → chunk → embed → vector-store write → retrieve, with vector-only recall, BM25+vector RRF hybrid retrieval, and optional LLM rerank. Embedding, vector-store, and rerank backends are all switchable via env variables between offline defaults and persistent backends. The BM25 keyword index and the vector store share one backend: either **SQLite + LanceDB** (offline default) or **PostgreSQL + pgvector** (via `VECTOR_STORE=pgvector`), so retrieval works immediately after restart with no rebuild.

## Why Rust

- **Errors are typed.** `BackendError` (src/error.rs) separates upstream failures (`Ollama`, `Rerank`) from internal ones (`VectorStore`, `KeywordIndex`), and the mapping to HTTP status codes is locked in `impl From<BackendError> for ApiError`: upstream-unreachable → 503, internal failures → 500. The runtime cannot ever pick the wrong status code, and a new backend can't be added without explicitly resolving its error path at compile time.
- **No illegal states.** `DocumentStatus`, `TaskStatus`, and `Priority` are enums (src/model.rs); a typo like `"reay"` simply does not compile. Every `match` over them is exhaustive — adding a variant forces the compiler to walk you through every site that must handle it (`as_str`, `parse`, SQL `status` column).
- **Ambiguity lives in the type.** Optional input is `Option<T>` (`limit: Option<usize>`, `mode: Option<SearchMode>`), so "not provided" and "0" are distinct and cannot be confused by callers.
- **The toolchain enforces the floor.** `cargo clippy -- -D warnings`, `cargo fmt --check`, and the unit-test suite run via `make verify`; a warning is a build failure, and the error-checking wins compounded across the whole pipeline.

## Quick start

```bash
make run
```

The service listens on `http://127.0.0.1:3000` by default; override with `PORT`:

```bash
make run PORT=8080
```

### Docker (two containers: nginx frontend + backend)

```bash
make docker-build   # build both images (first build takes a few minutes)
make docker-run     # build & start in background
make docker-logs    # follow container logs
make docker-stop    # stop & remove containers (data preserved)
```

- Frontend (nginx serving the built React app): `http://127.0.0.1:8080`
- Backend API (direct): `http://127.0.0.1:3000`

The frontend container reverses-proxies `/api` and `/health` to the backend on the compose network and injects the `x-api-key` header from `.env`, so browser `EventSource` works without special handling. The compose file also runs a `postgres` service (pgvector image) on `127.0.0.1:5433`; with `VECTOR_STORE=pgvector` the backend connects to it via `postgres://ws:ws@postgres:5432/ws` inside the network. SQLite and LanceDB data persist to the named volume `app-data`; PostgreSQL data to `pg-data`. `.env` configuration (API key, vector store, etc.) is injected into both containers automatically.

See all available commands:

```bash
make help
```

## Environment configuration

Copy the example config and edit:

```bash
cp .env.example .env
```

### Generating an API key

```bash
# openssl (recommended)
openssl rand -hex 32

# Python
python3 -c "import secrets; print(secrets.token_hex(32))"

# Node.js
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

Put the key in `.env`:

```bash
API_KEY=your-otp-generated-key-here
```

> **Note**: `.env` is gitignored. Keep the API key safe in production.

## Configuration

All via env variables; with defaults, the service runs fully offline (hash pseudo-embeddings + in-memory vector store) with zero external deps:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `3000` | Service listen port |
| `DATABASE_PATH` | `data/app.db` | SQLite database path; tasks, document metadata, and events persist here |
| `VECTOR_STORE` | `in_memory` | Vector backend: `in_memory`, `lancedb` (local Lance files), or `pgvector` (PostgreSQL; requires `DATABASE_URL`) |
| `LANCEDB_PATH` | `data/lancedb` | LanceDB data directory (only when `VECTOR_STORE=lancedb`) |
| `DATABASE_URL` | _empty_ | PostgreSQL connection string (required when `VECTOR_STORE=pgvector`); used for both the vector store and the keyword index |
| `EMBEDDING_API_BASE` | _empty_ | Embedding API endpoint; set to use real semantic embeddings, empty to use local hash pseudo-embeddings (FNV-1a bigram hash, stable across processes) |
| `EMBEDDING_API_KEY` | _empty_ | Embedding API auth token (not needed for local Ollama) |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `SEARCH_MODE` | `hybrid` | `hybrid` (BM25+vector RRF fusion) or `vector` (vector-only) |
| `RRF_K` | `60` | RRF fusion constant; larger smooths rank differences between the two legs |
| `RERANK_MODEL` | _empty_ | Rerank model name (e.g. `llama3.1:8b`); enables LLM rerank when set |
| `RERANK_API_BASE` | _empty_ | Rerank API endpoint; falls back to `EMBEDDING_API_BASE` |
| `RERANK_TOP_N` | `5` | Candidates sent to rerank |

### Local Ollama for real semantic embeddings

```bash
ollama pull nomic-embed-text
ollama serve
EMBEDDING_API_BASE=http://127.0.0.1:11434/v1 EMBEDDING_MODEL=nomic-embed-text \
  VECTOR_STORE=lancedb make run
```

Ollama's `/v1/embeddings` is OpenAI-compatible, so any compatible endpoint works. `OllamaEmbedder` probes the output dimension at construction and names the LanceDB table by it (`chunks_d{dim}`), so switching models can't silently write wrong-dimension vectors.

### Persistent vector store (LanceDB)

With `VECTOR_STORE=lancedb`, vectors are written to `LANCEDB_PATH` and searchable after restart; tasks, document metadata, and events persist in the SQLite file at `DATABASE_PATH` and fully recover. The BM25 keyword index (postings and document-length stats) also lives in SQLite (`chunk_meta` / `chunk_terms` tables), so no full rebuild at startup — hybrid search works immediately after restart.

> Building: local builds need Rust ≥ 1.91; LanceDB indirectly depends on `lzma-sys`, so on a minimal environment install `xz` (macOS: `brew install xz`) and `protoc` (the prost build script needs it; macOS: `brew install protobuf`) first. The Dockerfile already installs `protobuf-compiler` + `libprotobuf-dev` (the well-known `.proto` files live in the latter), so `make docker-build` just works.

### PostgreSQL + pgvector for scale

`VECTOR_STORE=pgvector` moves both the vector store and the BM25 keyword index into a single PostgreSQL database with the `vector` extension (as `chunk_meta` / `chunk_terms` / `chunks_d{dim}` tables). It is the target backend for tens-of-thousands to hundreds-of-thousands of documents, where an in-process file-based store starts to degrade. On startup the service idempotently creates the extension, tables, and an HNSW index (`vector_cosine_ops`) on the chunk table.

```bash
docker compose up -d postgres            # compose includes the pgvector/postgres service
VECTOR_STORE=pgvector \
  DATABASE_URL=postgres://ws:ws@127.0.0.1:5433/ws \
  make run
```

> When running inside compose, point `DATABASE_URL` at the service name instead: `postgres://ws:ws@postgres:5432/ws` (injected automatically from `.env`). Pick the dot product / cosine / L2 distance metric via the vector-similarity metric; the current implementation uses cosine (`<=>`).

### Performance at scale

BM25 ranking is executed **inside PostgreSQL**: a single SQL statement (a `WITH` stats/df CTE join) computes the full BM25 score (`k1 = 1.2`, `b = 0.75`) per chunk and applies the top-N `LIMIT`, so the driver never pulls postings into the application. Vector search rides the HNSW index with constant latency. Release-build results on a synthetic corpus (~17 chunks / 8 KB doc, 20k-doc scale ≈ 340k chunks):

| chunks  | hybrid (BM25 pushed down) | vector (HNSW) |
| ------- | ------------------------- | ------------- |
| 17k     | 54 ms                     | <2.4 ms       |
| 51k     | 134 ms                    | <2.4 ms       |
| 102k    | 426 ms                    | <2.4 ms       |
| 170k    | 580 ms                    | <2.4 ms       |
| 255k    | 939 ms                    | <2.4 ms       |
| 340k    | 1.5 s                     | <2.4 ms       |

Before the SQL push-down, per-doc postings were fetched to the application and re-scored in Rust; that path was 1306 ms at 17k chunks and 4.4 s at 340k chunks — push-down is ~2.9–24× faster. LanceDB on the same corpus degraded past roughly 30–50k chunks (51k: vector 130 ms, hybrid 617 ms). Rule of thumb: LanceDB is fine at low tens of thousands of chunks; switch to `VECTOR_STORE=pgvector` beyond that.

> Caveat: the benchmark corpus has only 38 distinct words, so every term matches ~320k chunks — a pathological hot-term case. Real corpora with richer distributions fare much better; a single-document filter keeps hybrid at 31–53 ms at any scale. Scoring parity: against the file-based/SQLite index, PG hybrid returns identical order within filtered sets (only RRF tie ordering differs on the full corpus).

### Chunking & document-metadata cache

Documents are split by Markdown `#` headings, then chunked within each section by 800-char windows with 120-char overlap; untitled plain text falls back to whole-note windows. Each chunk prefixes its heading ancestor chain (e.g. `## LanceDB`) into the text, so semantic retrieval and BM25 both hit "title-as-answer" queries. Document names for search results come from an in-process cache first; only cache-miss IDs hit SQLite — no per-search document-table scan.

## Retrieval: hybrid search & rerank

`/api/search` supports two recall modes (controlled by `SEARCH_MODE` or the `mode` param):

- **`vector`**: pure vector recall via nearest-neighbor search in the configured vector store (LanceDB or pgvector).
- **`hybrid`** (default): BM25 keyword retrieval (`jieba-rs` Chinese tokenizer) and vector retrieval run **concurrently**, then RRF (Reciprocal Rank Fusion, constant `RRF_K`). RRF raw scores are normalized to `[0,1]` to keep frontend percentage semantics. Hybrid works especially well on short Chinese queries — it fills the lexical gap embeddings miss.

**Metadata document-scoped filters**: the `documents` param takes a comma-separated list of document IDs; retrieval (both legs) only recalls chunks of those documents. The predicate is pushed down (`document_id IN (...)`) on the vector side to LanceDB or pgvector, and skipped outside the scope on the BM25 side, keeping both legs consistent. Invalid IDs return `400`.

Optional **LLM rerank** (enabled by `RERANK_MODEL`): takes the top `RERANK_TOP_N` fused candidates, reranks listwise, then truncates by relevance. Rerank is an **optional enhancement** — if the LLM call fails (timeout / network / unparseable output), it degrades to the original order, so search always returns results instead of erroring.

> Performance reference (local Ollama, debug build): vector / hybrid single search ~90ms; with `llama3.1:8b` rerank ~5–11s per query. Rerank concurrency is limited by Ollama single-stream processing; for production scale up `OLLAMA_NUM_PARALLEL` or use a dedicated rerank service.

## Verification

Run the full quality check (offline; default hash embeddings + in-memory vector store):

```bash
make verify
```

API smoke test (task create, optimistic-lock update, events, delete):

```bash
make api-smoke
```

RAG smoke test (upload, index, retrieve; default LanceDB + hash embeddings, auto Ollama when `EMBEDDING_API_BASE` is set):

```bash
make rag-smoke
```

Retrieval evaluation (`fixtures/eval.json` ships a query set annotated with expected documents; spins up a temp environment, uploads the sample documents, compares vector / hybrid recall and MRR, and verifies metadata filters don't leak. Non-zero exit when any mode's recall@k is below `EVAL_MIN_RECALL` (default 1.0) or a filter leaks in):

```bash
make eval
EVAL_MIN_RECALL=0.7 make eval   # relax threshold
```

## React frontend

The frontend lives in `frontend/` and proxies `/api` and `/health` to the Axum service via the Vite dev server. First-time install:

```bash
make frontend-install
```

Start backend and frontend with one command; `Ctrl+C` stops both:

```bash
make dev
```

Then open `http://127.0.0.1:5173`. The page has four sections:

- **Task console**: board, stats cards, event timeline. "Add test task" creates three sample tasks and advances them.
- **RAG document retrieval**: upload `.txt`/`.md`/`.markdown` docs, watch the import task progress and document status transitions (queued → processing → ready), then search semantically and see scored results.
- **Event live stream**: via SSE (`/api/events/stream`) task & document events are pushed; import events refresh the document list instantly (auto-degrade to 2s polling when SSE disconnects).
- **How it works**: ends-to-end pipeline walkthrough.

Production build:

```bash
make frontend-build
```

## API

- `GET /`: service info
- `GET /health`: health check & task count
- `GET /api/tasks`: paginated list; `status`, `offset`, `limit`
- `POST /api/tasks`: create task
- `GET /api/tasks/{id}`: fetch one task
- `PATCH /api/tasks/{id}`: update; must carry current `expected_version`
- `DELETE /api/tasks/{id}?expected_version=1`: delete task
- `GET /api/events`: retained events (max 200). `after` (incremental start id) & `limit` params
- `GET /api/events/stream`: SSE stream. On connect replays events after the `Last-Event-ID` request header (or the last 20 if absent), then pushes live task & document events (`data` = event JSON, `id` = event id, ~15s heartbeat). Browser EventSource auto-sends `Last-Event-ID` for resume. Auth must be injected by a reverse proxy `x-api-key` (browser EventSource can't set headers; the vite dev proxy already does; in Docker the nginx frontend container injects `.env` key)
- `GET /api/stats`: task status stats
- `POST /api/documents`: multipart text upload (`.txt`/`.md`/`.markdown`, ≤ 5 MiB), returns document & import task
- `GET /api/documents`: document list & status
- `GET /api/search?q=...&mode=hybrid&rerank=true&limit=5&documents=1,3`: returns matched chunks & scores. `mode` `hybrid` (default) / `vector`; `rerank=true` enables LLM rerank (needs `RERANK_MODEL`); `limit` caps count; `documents` scopes recall to those IDs

## Examples

```bash
# Create task
curl -X POST http://127.0.0.1:3000/api/tasks \
  -H 'content-type: application/json' \
  -d '{"title":"build concurrent task service","priority":"high"}'

# Update version 1. Concurrent clients holding a stale version get 409.
curl -X PATCH http://127.0.0.1:3000/api/tasks/1 \
  -H 'content-type: application/json' \
  -d '{"status":"in_progress","expected_version":1}'

# Upload doc & search
curl -X POST http://127.0.0.1:3000/api/documents -F "file=@notes.md"
curl http://127.0.0.1:3000/api/documents
curl "http://127.0.0.1:3000/api/search?q=safe+concurrency"

# Events & stats
curl http://127.0.0.1:3000/api/events
curl http://127.0.0.1:3000/api/stats
```

<div align="center">[🇨🇳 中文文档](README.zh.md)</div>