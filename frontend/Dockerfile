# ---------- 构建阶段：编译 React 静态产物 ----------
FROM node:22-bookworm-slim AS builder
WORKDIR /app
# 锁定依赖版本，构建可复现
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---------- 运行阶段：nginx 托管静态文件并反代后端 ----------
FROM nginx:1.27-alpine
# nginx 官方镜像支持 /etc/nginx/templates/*.template 经 envsubst 展开；
# ${API_KEY} 在启动时被替换为环境变量，用于给 /api 注入 x-api-key。
COPY frontend/nginx-default.conf.template /etc/nginx/templates/default.conf.template
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD ["wget", "-q", "-O", "-", "http://127.0.0.1/health"]
