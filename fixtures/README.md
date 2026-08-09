# RAG 测试样例文档

用于验证语义检索（RAG）流水线的样例文本文档。三个文档分属互不相关的主题，便于检验检索能否按查询命中对应领域：

| 文件 | 主题 | 典型查询 |
|------|------|----------|
| `doc-rust.md` | Rust 异步编程 | 并发安全、async runtime |
| `doc-ml.md` | 机器学习基础 | 神经网络训练、梯度下降 |
| `doc-cook.md` | 红烧肉做法 | 炖肉放什么调料、糖色 |

## 上传方式

启动服务后，逐个上传到 `/api/documents`：

```bash
for f in fixtures/doc-rust.md fixtures/doc-ml.md fixtures/doc-cook.md; do
  curl -X POST http://127.0.0.1:3000/api/documents -F "file=@$f"
done
```

或直接在前端 RAG 面板的「上传文档」处选择文件。

## 说明

embedding 模型对中文检索质量影响极大：

- `nomic-embed-text`（默认，768 维）：以英文为主，中文查询区分度极弱，
  三个文档得分挤在 0.62 窄带，红烧肉类查询会误排到 Rust 文档。
- `snowflake-arctic-embed2`（1024 维，多语言，**已验证**）：中文区分度优秀，
  搜「红烧肉」得 53% / 10% / 4% 的清晰三档分离，正确命中烹饪文档。

切换多语言模型只需重启时改环境变量，代码无需改动：

```bash
VECTOR_STORE=lancedb \
EMBEDDING_API_BASE=http://127.0.0.1:11434/v1 \
EMBEDDING_MODEL=snowflake-arctic-embed2:latest \
cargo run
```

> 注意：切换模型会改变向量维度，必须清空旧向量库（`rm -rf data/lancedb`）后重新上传文档，
> 否则 LanceDB 表维度不匹配会报错。`bge-m3` 同样适用此方式。
