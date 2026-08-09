.DEFAULT_GOAL := help

CARGO ?= cargo
PORT ?= 3000
RUST_LOG ?= info
SERVER_URL := http://127.0.0.1:$(PORT)

.PHONY: help \
	run run-release dev watch build build-release check \
	test test-unit test-doc smoke api-smoke rag-smoke rag-hybrid-smoke rag-fixtures sse-smoke eval \
	fmt fmt-check lint verify ci docs docs-open tree update \
	install-dev-tools frontend-install frontend frontend-build stop clean \
	gen-api-key docker-build docker-run docker-logs docker-stop

help: ## 显示可用命令
	@awk 'BEGIN {FS = ":.*##"}; /^[a-zA-Z_-]+:.*##/ {printf "%-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

run: ## 以开发配置启动服务（PORT=3000，RUST_LOG=info）
	PORT=$(PORT) RUST_LOG=$(RUST_LOG) $(CARGO) run

run-release: build-release ## 以发布配置启动服务
	PORT=$(PORT) RUST_LOG=$(RUST_LOG) $(CARGO) run --release

stop: ## 停止占用后端/前端端口的遗留进程（3000/5173）
	@-lsof -ti:$(PORT) -ti:5173 2>/dev/null | xargs kill -9 2>/dev/null || true

dev: stop ## 同时启动 Axum 和 React 开发服务器（后端就绪后再起前端，Ctrl+C 一并停止）
	@$(MAKE) run & backend_pid=$$!; \
	echo "==> waiting for backend on $(SERVER_URL) ..."; \
	for attempt in $$(seq 1 30); do \
		curl --fail --silent "$(SERVER_URL)/health" >/dev/null 2>&1 && { echo "==> backend ready"; break; }; \
		sleep 1; \
	done; \
	$(MAKE) frontend & frontend_pid=$$!; \
	trap 'kill $$backend_pid $$frontend_pid 2>/dev/null || true' EXIT INT TERM; \
	wait $$backend_pid; \
	wait $$frontend_pid

watch: ## 监听代码变更并自动重启服务（需要 cargo-watch）
	PORT=$(PORT) RUST_LOG=$(RUST_LOG) $(CARGO) watch -x run

build: ## 构建开发版本
	$(CARGO) build

build-release: ## 构建优化后的发布版本
	$(CARGO) build --release

check: ## 快速检查编译错误
	$(CARGO) check --all-targets

test: test-unit test-doc ## 运行全部测试

test-unit: ## 运行单元和集成测试
	$(CARGO) test --all-targets

test-doc: ## 构建文档以校验文档注释
	$(CARGO) doc --no-deps

smoke: ## 启动服务后验证健康检查和首页接口
	@PORT=$(PORT) RUST_LOG=$(RUST_LOG) $(CARGO) run --quiet & server_pid=$$!; \
	trap 'kill $$server_pid 2>/dev/null || true' EXIT INT TERM; \
	for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do \
		curl --fail --silent "$(SERVER_URL)/health" >/dev/null && break; \
		sleep 1; \
	done; \
	curl --fail --silent "$(SERVER_URL)/health"; echo; \
	curl --fail --silent "$(SERVER_URL)/"; echo

api-smoke: ## 验证任务的创建、更新、事件和删除流程
	@rm -f /tmp/ws-api-smoke.db; \
	[ -f .env ] && . ./.env; \
	set -e; \
	AUTH="$${API_KEY:+x-api-key: $$API_KEY}"; \
	PORT=$(PORT) RUST_LOG=$(RUST_LOG) DATABASE_PATH=/tmp/ws-api-smoke.db $(CARGO) run --quiet & server_pid=$$!; \
	trap 'kill $$server_pid 2>/dev/null || true' EXIT INT TERM; \
	for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do \
		curl --fail --silent "$(SERVER_URL)/health" >/dev/null && break; \
		sleep 1; \
	done; \
	task=$$(curl --fail --silent $${AUTH:+-H "$$AUTH"} -X POST "$(SERVER_URL)/api/tasks" -H 'content-type: application/json' -d '{"title":"api smoke task","priority":"high"}'); \
	task_id=$$(printf '%s' "$$task" | sed -E 's/.*"id":([0-9]+).*/\1/'); \
	test "$$task_id" -gt 0; \
	updated=$$(curl --fail --silent $${AUTH:+-H "$$AUTH"} -X PATCH "$(SERVER_URL)/api/tasks/$$task_id" -H 'content-type: application/json' -d '{"status":"in_progress","expected_version":1}'); \
	printf '%s' "$$updated" | grep -q '"version":2'; \
	printf '%s' "$$updated" | grep -q '"status":"in_progress"'; \
	curl --fail --silent $${AUTH:+-H "$$AUTH"} "$(SERVER_URL)/api/events" | grep -q 'task.updated'; \
	curl --fail --silent $${AUTH:+-H "$$AUTH"} -X DELETE "$(SERVER_URL)/api/tasks/$$task_id?expected_version=2" >/dev/null; \
		echo "API smoke test passed."

rag-smoke: ## 验证 RAG 上传、索引与检索（默认 LanceDB+hash embedding；配置 EMBEDDING_API_BASE 走 Ollama）
	@rm -rf /tmp/ws-rag-smoke-lancedb; \
	rm -f /tmp/ws-rag-smoke.db; \
	[ -f .env ] && . ./.env; \
	set -e; \
	AUTH="$${API_KEY:+x-api-key: $$API_KEY}"; \
	PORT=$(PORT) RUST_LOG=$(RUST_LOG) VECTOR_STORE=lancedb LANCEDB_PATH=/tmp/ws-rag-smoke-lancedb DATABASE_PATH=/tmp/ws-rag-smoke.db $(CARGO) run --quiet & server_pid=$$!; \
	trap 'kill $$server_pid 2>/dev/null || true' EXIT INT TERM; \
	for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do \
		curl --fail --silent "$(SERVER_URL)/health" >/dev/null 2>&1 && break; \
		sleep 1; \
	done; \
	printf '# Rust\nRust provides ownership and safe concurrency. Tokio powers async services.\n' > /tmp/ws-rag-smoke.md; \
	curl --fail --silent $${AUTH:+-H "$$AUTH"} -X POST "$(SERVER_URL)/api/documents" -F "file=@/tmp/ws-rag-smoke.md" >/dev/null; \
	for attempt in 1 2 3 4 5 6 7 8 9 10; do \
		docs=$$(curl --fail --silent $${AUTH:+-H "$$AUTH"} "$(SERVER_URL)/api/documents"); \
		echo "$$docs" | grep -q '"status":"ready"' && break; \
		echo "$$docs" | grep -q '"status":"failed"' && { echo "document import failed"; exit 1; }; \
		sleep 0.5; \
	done; \
	curl --fail --silent $${AUTH:+-H "$$AUTH"} "$(SERVER_URL)/api/documents" | grep -q '"status":"ready"'; \
	curl --fail --silent $${AUTH:+-H "$$AUTH"} "$(SERVER_URL)/api/search?q=safe+concurrency&limit=3" | grep -q '"text"'; \
	echo "RAG smoke test passed."; \
	rm -f /tmp/ws-rag-smoke.md

sse-smoke: ## 验证 SSE 事件流：回放 + Last-Event-ID 断线续传 + 实时 task.created
	@rm -f /tmp/ws-sse-smoke.db; \
	[ -f .env ] && . ./.env; \
	set -e; \
	AUTH="$${API_KEY:+x-api-key: $$API_KEY}"; \
	PORT=$(PORT) RUST_LOG=$(RUST_LOG) DATABASE_PATH=/tmp/ws-sse-smoke.db $(CARGO) run --quiet & server_pid=$$!; \
	trap 'kill $$server_pid 2>/dev/null || true' EXIT INT TERM; \
	for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do \
		curl --fail --silent "$(SERVER_URL)/health" >/dev/null && break; \
		sleep 1; \
	done; \
	curl --fail --silent $${AUTH:+-H "$$AUTH"} -X POST "$(SERVER_URL)/api/tasks" -H 'content-type: application/json' -d '{"title":"sse smoke pre"}' >/dev/null; \
	first_id=$$(curl --fail --silent $${AUTH:+-H "$$AUTH"} "$(SERVER_URL)/api/events" | python3 -c 'import sys,json;print(json.load(sys.stdin)["latest_event_id"])'); \
	echo "baseline event id=$$first_id"; \
	curl --fail --silent -N $${AUTH:+-H "$$AUTH"} -H "Last-Event-ID: $$first_id" "$(SERVER_URL)/api/events/stream" > /tmp/ws-sse-smoke.out & sse_pid=$$!; \
	sleep 1; \
	curl --fail --silent $${AUTH:+-H "$$AUTH"} -X POST "$(SERVER_URL)/api/tasks" -H 'content-type: application/json' -d '{"title":"sse resume target"}' >/dev/null; \
	sleep 1.5; \
	kill $$sse_pid 2>/dev/null || true; \
	next_id=$$((first_id + 1)); \
	grep -q "\"id\":$$next_id" /tmp/ws-sse-smoke.out; \
	if grep -q "\"id\":$$first_id" /tmp/ws-sse-smoke.out; then echo "baseline event replayed"; exit 1; fi; \
	echo "SSE smoke test passed (resume from id=$$first_id, got id=$$next_id, no replay)."; \
	rm -f /tmp/ws-sse-smoke.out /tmp/ws-sse-smoke.db

rag-fixtures: ## 将 fixtures/ 下的样例文档上传到运行中的服务（先 make run 启动后端）
	@for f in fixtures/*.md; do \
		case "$$f" in */README.md) continue;; esac; \
		printf 'uploading %s ... ' "$$f"; \
		curl --fail --silent -X POST "$(SERVER_URL)/api/documents" -F "file=@$$f" >/dev/null && echo ok || echo "FAILED"; \
	done

eval: ## 起临时环境、上传样例文档并跑检索评估（vector vs hybrid + metadata 过滤）
	@rm -rf /tmp/ws-eval-lancedb; \
	rm -f /tmp/ws-eval.db; \
	[ -f .env ] && . ./.env; \
	set -e; \
	AUTH="$${API_KEY:+x-api-key: $$API_KEY}"; \
	PORT=$(PORT) RUST_LOG=$(RUST_LOG) VECTOR_STORE=lancedb LANCEDB_PATH=/tmp/ws-eval-lancedb DATABASE_PATH=/tmp/ws-eval.db $(CARGO) run --quiet & server_pid=$$!; \
	trap 'kill $$server_pid 2>/dev/null || true' EXIT INT TERM; \
	for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do \
		curl --fail --silent "$(SERVER_URL)/health" >/dev/null 2>&1 && break; \
		sleep 1; \
	done; \
	for f in fixtures/doc-rust.md fixtures/doc-ml.md fixtures/doc-cook.md; do \
		curl --fail --silent $${AUTH:+-H "$$AUTH"} -X POST "$(SERVER_URL)/api/documents" -F "file=@$$f" >/dev/null; \
	done; \
	for attempt in 1 2 3 4 5 6 7 8 9 10; do \
		docs=$$(curl --fail --silent $${AUTH:+-H "$$AUTH"} "$(SERVER_URL)/api/documents"); \
		echo "$$docs" | grep -q '"status":"failed"' && { echo "document import failed"; exit 1; }; \
		echo "$$docs" | grep -c '"status":"ready"' | grep -q 3 && break; \
		sleep 0.5; \
	done; \
	python3 scripts/eval_retrieval.py --base "$(SERVER_URL)" $${API_KEY:+--key "$$API_KEY"} --min-recall $${EVAL_MIN_RECALL:-1.0}

rag-hybrid-smoke: ## 验证混合检索：对比 vector / hybrid 模式，并测试 rerank 开关（hash embedding，无需 Ollama）
	@rm -rf /tmp/ws-rag-hybrid-lancedb; \
	rm -f /tmp/ws-rag-hybrid.db; \
	[ -f .env ] && . ./.env; \
	set -e; \
	AUTH="$${API_KEY:+x-api-key: $$API_KEY}"; \
	PORT=$(PORT) RUST_LOG=$(RUST_LOG) VECTOR_STORE=lancedb LANCEDB_PATH=/tmp/ws-rag-hybrid-lancedb DATABASE_PATH=/tmp/ws-rag-hybrid.db $(CARGO) run --quiet & server_pid=$$!; \
	trap 'kill $$server_pid 2>/dev/null || true' EXIT INT TERM; \
	for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do \
		curl --fail --silent "$(SERVER_URL)/health" >/dev/null 2>&1 && break; \
		sleep 1; \
	done; \
	printf '# Rust\nRust ownership and borrowing guarantee memory safety and concurrency safety. Tokio powers async services.\n' > /tmp/ws-hybrid-rust.md; \
	printf '# Cooking\nBraised pork is a classic Chinese dish with soy sauce and sugar.\n' > /tmp/ws-hybrid-cook.md; \
	curl --fail --silent $${AUTH:+-H "$$AUTH"} -X POST "$(SERVER_URL)/api/documents" -F "file=@/tmp/ws-hybrid-rust.md" >/dev/null; \
	curl --fail --silent $${AUTH:+-H "$$AUTH"} -X POST "$(SERVER_URL)/api/documents" -F "file=@/tmp/ws-hybrid-cook.md" >/dev/null; \
	for attempt in 1 2 3 4 5 6 7 8 9 10; do \
		docs=$$(curl --fail --silent $${AUTH:+-H "$$AUTH"} "$(SERVER_URL)/api/documents"); \
		echo "$$docs" | grep -c '"status":"ready"' | grep -q 2 && break; \
		echo "$$docs" | grep -q '"status":"failed"' && { echo "document import failed"; exit 1; }; \
		sleep 0.5; \
	done; \
	echo '[vector mode]'; \
	curl --fail --silent $${AUTH:+-H "$$AUTH"} "$(SERVER_URL)/api/search?q=concurrency+ownership&mode=vector&limit=2" | grep -q '"text"'; \
	echo '[hybrid mode]'; \
	curl --fail --silent $${AUTH:+-H "$$AUTH"} "$(SERVER_URL)/api/search?q=concurrency+ownership&mode=hybrid&limit=2" | grep -q '"text"'; \
	echo '[default mode (hybrid)]'; \
	curl --fail --silent $${AUTH:+-H "$$AUTH"} "$(SERVER_URL)/api/search?q=concurrency+ownership&limit=2" | grep -q '"text"'; \
	echo '[hybrid chinese]'; \
	curl --fail --silent $${AUTH:+-H "$$AUTH"} "$(SERVER_URL)/api/search?q=内存安全&mode=hybrid&limit=2" | grep -q '"text"'; \
	echo '[rerank=false explicit]'; \
	curl --fail --silent $${AUTH:+-H "$$AUTH"} "$(SERVER_URL)/api/search?q=concurrency+ownership&mode=hybrid&rerank=false&limit=2" | grep -q '"text"'; \
	echo "Hybrid search smoke test passed."; \
	rm -f /tmp/ws-hybrid-rust.md /tmp/ws-hybrid-cook.md

fmt: ## 自动格式化代码
	$(CARGO) fmt

fmt-check: ## 检查代码格式
	$(CARGO) fmt -- --check

lint: ## 使用 Clippy 执行严格静态检查
	$(CARGO) clippy --all-targets -- -D warnings

verify: fmt-check check test lint ## 执行本地完整质量检查

ci: verify ## CI 入口：执行所有质量检查

docs: test-doc ## 生成 API 文档

docs-open: docs ## 在浏览器中打开已生成的 API 文档
	$(CARGO) doc --no-deps --open

tree: ## 显示依赖树
	$(CARGO) tree

update: ## 更新依赖锁定版本
	$(CARGO) update

install-dev-tools: ## 安装开发热重载工具 cargo-watch
	$(CARGO) install cargo-watch

frontend-install: ## 安装 React 前端依赖
	npm --prefix frontend install

frontend: ## 启动 React 开发服务器（默认端口：5173；自动加载根目录 .env 中的 API_KEY）
	@set -a; [ -f .env ] && . ./.env; set +a; \
	npm --prefix frontend run dev

frontend-build: ## 构建 React 前端生产文件
	npm --prefix frontend run build

clean: ## 清除构建产物
	$(CARGO) clean

gen-api-key: ## 重新生成 API_KEY 并写入 .env（同时更新 .env.example；旧键立即作废）
	@[ -f .env ] || { echo "缺少 .env，请先 cp .env.example .env"; exit 1; }; \
	key=$$(openssl rand -hex 32); \
	if grep -q '^API_KEY=' .env; then \
		perl -i -pe "s/^API_KEY=.*/API_KEY=$$key/" .env; \
	else \
		printf 'API_KEY=%s\n' "$$key" >> .env; \
	fi; \
	echo "已生成新的 API_KEY 并写入 .env（.gitignore 已忽略，不会入库）"; \
	echo "如需同步到 Docker 部署：在 compose 环境或部署密钥管理中注入该值"

docker-build: ## 构建容器镜像
	docker build -t rust-rag-task-service:latest .

docker-run: ## 用 Docker Compose 启动服务（默认 http://127.0.0.1:3000，数据持久化到命名卷）
	docker compose up -d

docker-logs: ## 跟踪容器日志
	docker compose logs -f

docker-stop: ## 停止并移除容器（保留数据卷）
	docker compose down
