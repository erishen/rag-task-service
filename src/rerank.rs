use std::collections::HashSet;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;

use crate::chunk::ChunkMatch;
use crate::error::BackendError;

/// 喂给 LLM 的单条候选文本截断长度，避免超出上下文。
const CANDIDATE_TEXT_LIMIT: usize = 500;

/// 重排器：对召回结果按 query 相关性二次排序，返回 top-k。
#[async_trait]
pub(crate) trait Reranker: Send + Sync {
    async fn rerank(
        &self,
        query: &str,
        candidates: Vec<ChunkMatch>,
        top_k: usize,
    ) -> Result<Vec<ChunkMatch>, BackendError>;
}

/// 未配置重排时的空实现：按原顺序截断 top_k，零延迟。
/// 生产路径用 `Option::None` 表示不重排，此实现保留为测试替身。
#[allow(dead_code)]
pub(crate) struct NoopReranker;

#[async_trait]
impl Reranker for NoopReranker {
    async fn rerank(
        &self,
        _query: &str,
        mut candidates: Vec<ChunkMatch>,
        top_k: usize,
    ) -> Result<Vec<ChunkMatch>, BackendError> {
        candidates.truncate(top_k);
        Ok(candidates)
    }
}

/// 通过 Ollama `/chat/completions`（OpenAI 兼容）做 listwise 重排。
/// 仅网络层失败返回 Err（503）；LLM 输出解析失败时回退原顺序，保证可用性。
pub(crate) struct LlmReranker {
    client: reqwest::Client,
    endpoint: String,
    api_key: Option<String>,
    model: String,
    top_n: usize,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Deserialize)]
struct ChatMessage {
    content: String,
}

impl LlmReranker {
    pub(crate) fn new(base: &str, api_key: Option<String>, model: String, top_n: usize) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(15))
            .build()
            .map_err(|error| BackendError::rerank(error.to_string()))
            .expect("failed to build rerank http client");
        Self {
            client,
            endpoint: format!("{}/chat/completions", base.trim_end_matches('/')),
            api_key,
            model,
            top_n,
        }
    }

    async fn call_llm(
        &self,
        query: &str,
        candidates: &[ChunkMatch],
    ) -> Result<String, BackendError> {
        let list = candidates
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let text = truncate_text(&m.text, CANDIDATE_TEXT_LIMIT);
                format!("{i}. {text}")
            })
            .collect::<Vec<_>>()
            .join("\n");
        let messages = serde_json::json!([
            {
                "role": "system",
                "content": "你是检索结果重排器。给定用户查询和候选文本块列表，按与查询的相关性从高到低排序。只输出一个 JSON 数组，元素为候选编号（从 0 开始的整数），不要输出任何其他内容。格式示例：[2,0,5,1,3,4]"
            },
            {
                "role": "user",
                "content": format!("查询：{query}\n候选：\n{list}")
            }
        ]);
        let mut builder = self.client.post(&self.endpoint).json(&serde_json::json!({
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 64,
            "stream": false,
        }));
        if let Some(key) = &self.api_key {
            builder = builder.bearer_auth(key);
        }
        let response = builder
            .send()
            .await
            .map_err(|error| BackendError::rerank(error.to_string()))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            eprintln!("rerank llm returned {status}: {body}");
            return Err(BackendError::rerank(format!(
                "rerank llm returned {status}"
            )));
        }
        let parsed: ChatResponse = response
            .json()
            .await
            .map_err(|error| BackendError::rerank(error.to_string()))?;
        parsed
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message.content)
            .ok_or_else(|| BackendError::rerank("rerank llm returned no choices"))
    }
}

#[async_trait]
impl Reranker for LlmReranker {
    async fn rerank(
        &self,
        query: &str,
        mut candidates: Vec<ChunkMatch>,
        top_k: usize,
    ) -> Result<Vec<ChunkMatch>, BackendError> {
        if candidates.len() > self.top_n {
            candidates.truncate(self.top_n);
        }
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let n = candidates.len();
        // LLM 调用失败（超时/网络/HTTP）时降级返回原顺序：重排是可选增强，不应让搜索 503。
        let content = match self.call_llm(query, &candidates).await {
            Ok(content) => content,
            Err(error) => {
                eprintln!("rerank degraded to original order: {error}");
                return Ok(candidates.into_iter().take(top_k).collect());
            }
        };
        // 输出解析失败同样回退原顺序。
        let ordered = match parse_order(&content, n) {
            Some(order) => reorder(candidates, &order),
            None => candidates,
        };
        Ok(ordered.into_iter().take(top_k).collect())
    }
}

fn truncate_text(text: &str, limit: usize) -> &str {
    match text.char_indices().nth(limit) {
        Some((idx, _)) => &text[..idx],
        None => text,
    }
}

/// 从 LLM 输出中提取首个 `[...]` 并解析为索引数组。校验：范围合法、无重复、不超长。
fn parse_order(content: &str, n: usize) -> Option<Vec<usize>> {
    let start = content.find('[')?;
    let end = content.rfind(']')?;
    if end <= start {
        return None;
    }
    let slice = &content[start..=end];
    let indices: Vec<usize> = serde_json::from_str(slice).ok()?;
    if indices.iter().any(|&i| i >= n) {
        return None;
    }
    let mut seen = HashSet::new();
    for &i in &indices {
        if !seen.insert(i) {
            return None;
        }
    }
    if indices.len() > n {
        return None;
    }
    Some(indices)
}

/// 按 LLM 给定的顺序重排；LLM 遗漏的候选按原顺序追加，避免丢结果。
fn reorder(candidates: Vec<ChunkMatch>, order: &[usize]) -> Vec<ChunkMatch> {
    let mut result: Vec<ChunkMatch> = order
        .iter()
        .filter_map(|&i| candidates.get(i).cloned())
        .collect();
    let ranked: HashSet<usize> = order.iter().copied().collect();
    for (i, m) in candidates.iter().enumerate() {
        if !ranked.contains(&i) {
            result.push(m.clone());
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(document_id: u64, text: &str) -> ChunkMatch {
        ChunkMatch {
            document_id,
            chunk_index: 0,
            text: text.to_owned(),
            score: 0.0,
        }
    }

    #[test]
    fn parses_valid_json_indices() {
        let order = parse_order("[2,0,1]", 3).unwrap();
        assert_eq!(order, vec![2, 0, 1]);
    }

    #[test]
    fn extracts_array_from_surrounding_text() {
        let order = parse_order("排序结果：[1,0,2] 完成", 3).unwrap();
        assert_eq!(order, vec![1, 0, 2]);
    }

    #[test]
    fn falls_back_on_invalid_json() {
        assert!(parse_order("我认为 2 最好", 3).is_none());
    }

    #[test]
    fn falls_back_on_out_of_range_index() {
        assert!(parse_order("[0,99,1]", 3).is_none());
    }

    #[test]
    fn falls_back_on_duplicate_index() {
        assert!(parse_order("[0,0,1]", 3).is_none());
    }

    #[test]
    fn reorder_appends_unranked_candidates() {
        let candidates = vec![m(1, "a"), m(2, "b"), m(3, "c")];
        // LLM 只排了第 1 个，其余应按原顺序追加。
        let ordered = reorder(candidates, &[1]);
        assert_eq!(ordered.len(), 3);
        assert_eq!(ordered[0].document_id, 2);
        assert_eq!(ordered[1].document_id, 1);
        assert_eq!(ordered[2].document_id, 3);
    }

    #[tokio::test]
    async fn noop_reranker_truncates_to_top_k() {
        let reranker = NoopReranker;
        let candidates = vec![m(1, "a"), m(2, "b"), m(3, "c")];
        let result = reranker.rerank("q", candidates, 2).await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].document_id, 1);
    }
}
