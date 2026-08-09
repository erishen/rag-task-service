import { useEffect, useRef, useState } from "react";

const docStatusLabel = {
  queued: "排队中",
  processing: "处理中",
  ready: "已就绪",
  failed: "失败"
};

const TAB_IDS = [
  { id: "documents", label: "文档管理" },
  { id: "events", label: "事件实时流" },
  { id: "api", label: "API 文档" },
  { id: "architecture", label: "技术架构" }
];

const eventKindLabel = {
  "task.created": "任务创建",
  "task.updated": "任务更新",
  "task.deleted": "任务删除",
  "rag.parsing": "解析文档",
  "rag.embedding": "向量化",
  "rag.indexed": "已索引",
  "rag.failed": "导入失败"
};

function formatTime(unixSeconds) {
  const date = new Date(unixSeconds * 1000);
  return date.toLocaleTimeString("zh-CN", { hour12: false });
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function request(path, options = {}) {
  const isFormData = options.body instanceof FormData;
  const response = await fetch(path, {
    ...options,
    headers: {
      ...(options.body && !isFormData ? { "content-type": "application/json" } : {}),
      ...options.headers
    }
  });

  if (response.status === 204) return null;

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    // 携带 HTTP 状态码，供调用方识别 429 等可重试错误。
    const error = new Error(payload?.message ?? `请求失败（${response.status}）`);
    error.status = response.status;
    throw error;
  }
  return payload;
}

function isPending(doc) {
  return doc.status === "queued" || doc.status === "processing";
}

export default function App() {
  const [activeTab, setActiveTab] = useState("documents");
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [ragError, setRagError] = useState("");
  const [searchInput, setSearchInput] = useState("");
  const [searchResults, setSearchResults] = useState(null);
  const [searching, setSearching] = useState(false);
  const fileInput = useRef(null);
  const folderInput = useRef(null);
  const refreshTimer = useRef(null);
  const [events, setEvents] = useState([]);
  const [streamConnected, setStreamConnected] = useState(false);

  async function refreshDocuments() {
    try {
      const page = await request("/api/documents");
      setDocuments(page.items ?? []);
    } catch (cause) {
      // 429 是瞬时限流（per-IP 10 req/s，SSE 也计入额度），自动恢复，
      // 静默跳过而非整页报错；后续轮询/事件会补齐列表。
      if (cause.status === 429) return;
      setRagError(cause.message);
    }
  }

  useEffect(() => {
    void refreshDocuments();
  }, []);

  // SSE 事件流：实时接收任务/文档导入事件，并据此即时刷新文档状态。
  // EventSource 无法自定义 Header，鉴权依赖 vite 代理注入 x-api-key。
  useEffect(() => {
    let source;
    try {
      source = new EventSource("/api/events/stream");
    } catch {
      return undefined;
    }
    source.onopen = () => setStreamConnected(true);
    source.onmessage = (message) => {
      let event;
      try {
        event = JSON.parse(message.data);
      } catch {
        return;
      }
      setEvents((prev) => [event, ...prev].slice(0, 30));
      // 文档导入类事件触发刷新文档列表，但用防抖合并（250ms）：
      // 一次导入会连发 parsing/embedding/indexed 多个事件，逐个 GET 会在
      // 文件夹批量上传时撞上服务端限流。合并后同批事件只打一次请求。
      if (event.kind?.startsWith("rag.")) {
        clearTimeout(refreshTimer.current);
        refreshTimer.current = setTimeout(() => void refreshDocuments(), 250);
      }
    };
    source.onerror = () => {
      setStreamConnected(false);
      // 断开自动重连由 EventSource 自身处理。
    };
    return () => source.close();
  }, []);

  // 兜底轮询：仅当 SSE 未连接且存在排队/处理中文档时启用，SSE 正常时由事件驱动刷新。
  useEffect(() => {
    if (streamConnected || !documents.some(isPending)) return;
    const timer = setInterval(() => void refreshDocuments(), 2000);
    return () => clearInterval(timer);
  }, [documents, streamConnected]);

  async function uploadFiles(files) {
    // 前端按扩展名过滤，避免后端逐个返回 400
    const allowed = files.filter((f) => /\.(txt|md|markdown)$/i.test(f.name));
    // 项目/系统常见文件静默排除，不进"已跳过"提示：git 配置、JSON、DS_Store 等
    const quiet = /(^|\/|\\)(\.DS_Store|\.gitignore|\.env.*|package-lock\.json)$|\.(json|yaml|yml|toml)$/i;
    const skippedNames = files
      .filter((f) => !allowed.includes(f))
      .map((f) => f.name)
      .filter((n) => !quiet.test(n));
    let skippedMessage = "";
    if (skippedNames.length > 0) {
      // 明确列出被跳过的文件名（如 .DS_Store 等隐藏文件），避免用户困惑"都是 md 怎么被跳过"。
      skippedMessage = `已跳过 ${skippedNames.length} 个非文本文件：${skippedNames.join("、")}`;
    }
    if (allowed.length === 0) {
      setRagError(skippedNames.length > 0 ? skippedMessage : "没有可上传的文件");
      return;
    }
    setUploading(true);
    setRagError("");
    setUploadProgress({ total: allowed.length, done: 0, failed: 0, current: "" });
    let failed = 0;
    const errors = [];
    for (let i = 0; i < allowed.length; i++) {
      const f = allowed[i];
      setUploadProgress({ total: allowed.length, done: i, failed, current: f.name });
      let lastError = null;
      // 429 限流时按指数退避重试，最多重试 3 次，避免文件夹批量上传触发服务端限流。
      const maxAttempts = 4;
      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        try {
          const formData = new FormData();
          formData.append("file", f);
          await request("/api/documents", { method: "POST", body: formData });
          lastError = null;
          break;
        } catch (cause) {
          lastError = cause;
          if (cause.status === 429) {
            await sleep(400 * (attempt + 1));
            continue;
          }
          break;
        }
      }
      if (lastError) {
        failed += 1;
        errors.push(`${f.name}: ${lastError.message}`);
      }
      // 文件间节流，避免连发超过服务端每密钥限流（默认 10 req/s）。
      if (i < allowed.length - 1) await sleep(120);
    }
    setUploadProgress(null);
    if (skippedNames.length > 0) setRagError(skippedMessage);
    if (failed > 0) {
      const msg = `${failed} 个文件上传失败：${errors.join("；")}`;
      setRagError((prev) => (prev ? `${prev}；${msg}` : msg));
    }
    if (fileInput.current) fileInput.current.value = "";
    if (folderInput.current) folderInput.current.value = "";
    await refreshDocuments();
    setUploading(false);
  }

  async function uploadDocument(event) {
    event.preventDefault();
    const files = Array.from(fileInput.current?.files ?? []);
    if (files.length === 0) return;
    await uploadFiles(files);
  }

  async function uploadFolder(event) {
    event.preventDefault();
    const files = Array.from(folderInput.current?.files ?? []);
    if (files.length === 0) return;
    await uploadFiles(files);
  }

  async function resetDocuments() {
    if (!window.confirm("确定清空所有文档？\n\n将同时清空向量库与 BM25 索引的全部数据，此操作不可恢复。")) return;
    setUploading(true);
    setRagError("");
    try {
      await request("/api/documents", { method: "DELETE" });
      await refreshDocuments();
    } catch (cause) {
      setRagError(cause.message);
    } finally {
      setUploading(false);
    }
  }

  async function deleteDocument(id, name) {
    if (!window.confirm(`确定删除文档「${name}」？\n\n删除文档将同时清空该文档在向量库中的所有向量数据，此操作不可恢复。`)) return;
    setUploading(true);
    setRagError("");
    try {
      await request(`/api/documents/${id}`, { method: "DELETE" });
      await refreshDocuments();
    } catch (cause) {
      setRagError(cause.message);
    } finally {
      setUploading(false);
    }
  }

  async function runSearch(event) {
    event.preventDefault();
    if (!searchInput.trim()) return;
    setSearching(true);
    setRagError("");
    try {
      const response = await request(
        `/api/search?q=${encodeURIComponent(searchInput.trim())}&limit=6`
      );
      setSearchResults(response);
    } catch (cause) {
      setRagError(cause.message);
      setSearchResults(null);
    } finally {
      setSearching(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">AXUM · TOKIO · REACT · LANCEDB</p>
          <h1>RAG 文档检索系统</h1>
          <p className="subtitle">上传文档，自动切块、向量化，支持语义检索与混合搜索。</p>
        </div>
        <div className="header-actions">
          <button className="secondary" type="button" onClick={() => void refreshDocuments()} disabled={uploading}>
            刷新文档
          </button>
        </div>
      </header>

      {ragError && <p className="error" role="alert">{ragError}</p>}

      <nav className="tab-bar" role="tablist" aria-label="功能导航">
        {TAB_IDS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={activeTab === tab.id}
            className={`tab-button${activeTab === tab.id ? " active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
            {tab.id === "events" && (
              <span
                className={`tab-dot${streamConnected ? " on" : ""}`}
                title={streamConnected ? "SSE 已连接" : "SSE 未连接"}
              />
            )}
          </button>
        ))}
      </nav>

      {activeTab === "documents" && (
      <section className="panel rag-panel">
        <div className="section-heading">
          <div>
            <h2>文档管理</h2>
            <p>上传文本文档（.txt/.md），系统自动切块、向量化并写入向量库，支持语义检索。</p>
          </div>
        </div>

        <div className="rag-grid">
          <div className="rag-column">
            <h3>上传文档</h3>
            <div
              className={`rag-dropzone${dragActive ? " dragging" : ""}`}
              onDragOver={(event) => {
                event.preventDefault();
                setDragActive(true);
              }}
              onDragEnter={(event) => event.preventDefault()}
              onDragLeave={(event) => {
                event.preventDefault();
                if (event.currentTarget === event.target) setDragActive(false);
              }}
              onDrop={(event) => {
                event.preventDefault();
                setDragActive(false);
                const files = Array.from(event.dataTransfer.files ?? []);
                if (files.length > 0) void uploadFiles(files);
              }}
            >
              <input
                ref={fileInput}
                type="file"
                accept=".txt,.md,.markdown"
                multiple
                onChange={uploadDocument}
                aria-label="选择文档（可多选）"
                hidden
              />
              <input
                ref={(el) => {
                  folderInput.current = el;
                  if (el) {
                    el.setAttribute("webkitdirectory", "");
                    el.setAttribute("directory", "");
                  }
                }}
                type="file"
                multiple
                accept=".txt,.md,.markdown"
                onChange={uploadFolder}
                hidden
              />
              <p className="dz-title">把 .txt / .md 文档拖到这里</p>
              <p className="dz-hint">或点击下方按钮选择文件，支持多选与整目录上传</p>
              <div className="dz-actions">
                <button type="button" disabled={uploading} onClick={() => fileInput.current?.click()}>
                  选择文件
                </button>
                <button
                  type="button"
                  className="secondary"
                  disabled={uploading}
                  onClick={() => folderInput.current?.click()}
                >
                  上传文件夹
                </button>
              </div>
            </div>
            {uploadProgress && (
              <p className="upload-progress" role="status">
                上传中 {uploadProgress.done}/{uploadProgress.total}：{uploadProgress.current}
                {uploadProgress.failed > 0 ? `（已失败 ${uploadProgress.failed}）` : ""}
              </p>
            )}

            <h3>
              文档列表
              {documents.length > 0 && (
                <button
                  type="button"
                  className="btn-link-danger"
                  disabled={uploading}
                  onClick={resetDocuments}
                >
                  清空
                </button>
              )}
            </h3>
            {documents.length === 0 ? (
              <p className="empty">还没有文档，上传一个 .txt 或 .md 试试。</p>
            ) : (
              <ul className="doc-list">
                {documents.map((doc) => (
                  <li key={doc.id} className="doc-item">
                    <div className="doc-head">
                      <strong>{doc.name}</strong>
                      <div className="doc-actions">
                        <span className={`badge doc-status-${doc.status}`}>{docStatusLabel[doc.status] ?? doc.status}</span>
                        <button
                          type="button"
                          className="btn-link-danger"
                          disabled={uploading}
                          onClick={() => deleteDocument(doc.id, doc.name)}
                        >
                          删除
                        </button>
                      </div>
                    </div>
                    <span className="doc-meta">
                      {doc.chunk_count > 0 ? `${doc.chunk_count} 个文本块` : "尚未切分"} · {doc.source_chars} 字符
                    </span>
                    {doc.error && <span className="doc-error">失败：{doc.error}</span>}
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="rag-column">
            <h3>语义检索</h3>
            <form onSubmit={runSearch} className="rag-search">
              <input
                value={searchInput}
                onChange={(event) => setSearchInput(event.target.value)}
                placeholder="例如：safe concurrency"
                aria-label="检索关键词"
              />
              <button type="submit" disabled={searching || !searchInput.trim()}>
                {searching ? "检索中…" : "检索"}
              </button>
            </form>

            {searchResults && (
              searchResults.items.length === 0 ? (
                <p className="empty">没有匹配的文本块。</p>
              ) : (
                <ol className="search-results">
                  {searchResults.items.map((item, index) => (
                    <li key={`${item.document_id}-${item.chunk_index}-${index}`} className="search-result">
                      <div className="result-head">
                        <strong>{item.document_name}</strong>
                        <span className="result-score">{Math.round(item.score * 100)}% 匹配</span>
                      </div>
                      <p>{item.text}</p>
                      <span className="result-meta">文档 #{item.document_id} · 文本块 #{item.chunk_index}</span>
                    </li>
                  ))}
                </ol>
              )
            )}
          </div>
        </div>
      </section>
      )}

      {activeTab === "events" && (
      <section className="panel event-panel">
        <div className="section-heading">
          <div>
            <h2>事件实时流</h2>
            <p>通过 SSE（<code>/api/events/stream</code>）实时推送任务与文档导入事件。</p>
          </div>
          <span className={`stream-state${streamConnected ? " connected" : ""}`}>
            {streamConnected ? "● 已连接" : "○ 未连接"}
          </span>
        </div>
        {events.length === 0 ? (
          <p className="empty">暂无事件。上传文档或更新任务状态后会实时显示在这里。</p>
        ) : (
          <ul className="event-list">
            {events.map((event) => (
              <li key={event.id} className="event-item">
                <span className="badge event-kind">{event.kind}</span>
                <span className="event-label">{eventKindLabel[event.kind] ?? event.kind}</span>
                <span className="event-meta">任务 #{event.task_id} · v{event.task_version}</span>
                <span className="event-time">{formatTime(event.occurred_at)}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
      )}

      {activeTab === "api" && (
      <section className="panel api-panel">
        <div className="section-heading">
          <div>
            <h2>API 文档</h2>
            <p>对外提供的 RESTful API，可直接集成到你的应用中。</p>
          </div>
        </div>

        <div className="api-section">
          <h3>认证</h3>
          <div className="api-endpoint">
            <p>所有 API 请求需要在 Header 中携带 API Key（如果服务端配置了 <code>API_KEY</code> 环境变量）：</p>
            <div className="code-block">
              <pre>{`curl -H "x-api-key: your-secret-key" http://localhost:3000/api/documents`}</pre>
            </div>
            <div className="auth-note">
              <strong>注意：</strong>如果服务端未设置 <code>API_KEY</code> 环境变量，则跳过认证（适合本地开发）。
              生产环境强烈建议配置 API Key 以保护接口安全。
            </div>
          </div>
        </div>

        <div className="api-section">
          <h3>文档管理 API</h3>
          <div className="api-endpoint">
            <div className="endpoint-header">
              <span className="method post">POST</span>
              <code>/api/documents</code>
            </div>
            <p>上传文档并自动切块、向量化。支持 .txt/.md/.markdown 文件，最大 5MB。</p>
            <div className="code-block">
              <pre>{`curl -X POST http://localhost:3000/api/documents \\
  -H "x-api-key: your-secret-key" \\
  -F "file=@document.md"`}</pre>
            </div>
            <div className="response-example">
              <strong>响应示例：</strong>
              <pre>{`{
  "document": {
    "id": 1,
    "name": "document.md",
    "status": "queued",
    "source_chars": 1024,
    "chunk_count": 0
  },
  "task": {
    "id": 1,
    "title": "导入文档：document.md",
    "status": "queued"
  }
}`}</pre>
            </div>
          </div>

          <div className="api-endpoint">
            <div className="endpoint-header">
              <span className="method get">GET</span>
              <code>/api/documents</code>
            </div>
            <p>获取所有文档列表及其处理状态。</p>
            <div className="code-block">
              <pre>{`curl -H "x-api-key: your-secret-key" \\
  http://localhost:3000/api/documents`}</pre>
            </div>
          </div>

          <div className="api-endpoint">
            <div className="endpoint-header">
              <span className="method delete">DELETE</span>
              <code>/api/documents/{"{id}"}</code>
            </div>
            <p>删除指定文档及其所有向量数据。</p>
            <div className="code-block">
              <pre>{`curl -X DELETE -H "x-api-key: your-secret-key" \\
  http://localhost:3000/api/documents/1`}</pre>
            </div>
          </div>
        </div>

        <div className="api-section">
          <h3>语义检索 API</h3>
          <div className="api-endpoint">
            <div className="endpoint-header">
              <span className="method get">GET</span>
              <code>/api/search</code>
            </div>
            <p>语义检索文档内容。支持纯向量检索和混合检索（BM25+向量 RRF 融合）。</p>
            <div className="params-table">
              <table>
                <thead>
                  <tr>
                    <th>参数</th>
                    <th>类型</th>
                    <th>默认值</th>
                    <th>说明</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><code>q</code></td>
                    <td>string</td>
                    <td>必填</td>
                    <td>检索查询文本</td>
                  </tr>
                  <tr>
                    <td><code>mode</code></td>
                    <td>string</td>
                    <td>hybrid</td>
                    <td>检索模式：vector（纯向量）或 hybrid（混合）</td>
                  </tr>
                  <tr>
                    <td><code>limit</code></td>
                    <td>integer</td>
                    <td>5</td>
                    <td>返回结果数量</td>
                  </tr>
                  <tr>
                    <td><code>rerank</code></td>
                    <td>boolean</td>
                    <td>false</td>
                    <td>是否启用 LLM 重排（需配置 RERANK_MODEL）</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div className="code-block">
              <pre>{`# 混合检索（默认）
curl -H "x-api-key: your-secret-key" \\
  "http://localhost:3000/api/search?q=safe+concurrency&limit=5"

# 纯向量检索
curl -H "x-api-key: your-secret-key" \\
  "http://localhost:3000/api/search?q=safe+concurrency&mode=vector"

# 启用 LLM 重排
curl -H "x-api-key: your-secret-key" \\
  "http://localhost:3000/api/search?q=safe+concurrency&rerank=true"`}</pre>
            </div>
            <div className="response-example">
              <strong>响应示例：</strong>
              <pre>{`{
  "query": "safe concurrency",
  "items": [
    {
      "document_id": 1,
      "document_name": "rust-guide.md",
      "chunk_index": 0,
      "text": "Rust ownership and borrowing...",
      "score": 0.92
    }
  ]
}`}</pre>
            </div>
          </div>
        </div>

        <div className="api-section">
          <h3>服务端点</h3>
          <div className="api-endpoint">
            <div className="endpoint-header">
              <span className="method get">GET</span>
              <code>/health</code>
            </div>
            <p>健康检查，返回服务状态和任务数量。</p>
          </div>
        </div>
      </section>
      )}

      {activeTab === "architecture" && (
      <section className="panel explain-panel">
        <div className="section-heading">
          <div>
            <h2>技术架构</h2>
            <p>RAG 文档检索系统的端到端链路。</p>
          </div>
        </div>
        <ol className="flow">
          <li><strong>前端上传</strong><span>用户上传 .txt/.md 文档，前端过滤非文本文件。</span></li>
          <li><strong>后端接收</strong><span>Axum API 接收文档，创建导入任务并排队处理。</span></li>
          <li><strong>文本切块</strong><span>按段落和句子边界智能切分，保留语义完整性。</span></li>
          <li><strong>向量化</strong><span>使用 Embedding 模型将文本块转换为向量（支持本地 hash 或 Ollama 真实语义）。</span></li>
          <li><strong>存储索引</strong><span>向量写入 LanceDB，同时建立内存 BM25 索引用于混合检索。</span></li>
          <li><strong>语义检索</strong><span>支持纯向量检索、BM25+向量混合检索（RRF 融合）、以及可选的 LLM 重排。</span></li>
        </ol>
      </section>
      )}
    </main>
  );
}
