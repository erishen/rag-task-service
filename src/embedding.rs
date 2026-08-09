use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::BackendError;

const HASH_DIMENSIONS: usize = 384;
const OLLAMA_BATCH: usize = 64;

/// 把文本转向量。实现可以是本地 hash 伪 embedding，也可以是远程 Ollama。
#[async_trait]
pub(crate) trait Embedder: Send + Sync {
    fn dim(&self) -> usize;
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, BackendError>;
}

/// 本地 hash 伪 embedding：bigram 哈希到固定维度，离线可用，用于测试与默认 demo。
pub(crate) struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    pub(crate) fn new() -> Self {
        Self {
            dim: HASH_DIMENSIONS,
        }
    }

    fn embed_one(&self, text: &str) -> Vec<f32> {
        let normalized: Vec<char> = text
            .to_lowercase()
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect();
        let mut vector = vec![0.0; self.dim];
        for window in normalized.windows(2) {
            // 用稳定的 FNV-1a 而非 DefaultHasher：DefaultHasher 的哈希值 Rust 官方不保证
            // 跨版本/平台一致，会让持久化到 LanceDB 的旧向量在重启后检索错乱。
            let hash = fnv1a_bigram(window);
            let index = (hash as usize) % self.dim;
            vector[index] += if hash & 1 == 0 { 1.0 } else { -1.0 };
        }
        normalize(&mut vector);
        // LanceDB 的 cosine 距离对全零向量无定义，确保至少一个非零维度。
        if vector.iter().all(|value| *value == 0.0) {
            vector[0] = 1.0;
        }
        vector
    }
}

/// FNV-1a 64 位哈希。选择它而非 `DefaultHasher` 的原因：哈希结果在任意 Rust 版本与
/// 平台上都稳定，保证 hash embedding 落盘后跨进程一致（Rust 标准库只保证
/// `DefaultHasher` 的实现细节，不保证哈希值跨版本一致）。
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// 对两个字符组成的 bigram 做稳定哈希，输入为其 UTF-8 字节序列。
fn fnv1a_bigram(bigram: &[char]) -> u64 {
    let mut bytes = Vec::with_capacity(8);
    for character in bigram {
        let mut buffer = [0; 4];
        bytes.extend_from_slice(character.encode_utf8(&mut buffer).as_bytes());
    }
    fnv1a(&bytes)
}

impl Default for HashEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Embedder for HashEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, BackendError> {
        Ok(texts.iter().map(|text| self.embed_one(text)).collect())
    }
}

/// 通过 OpenAI 兼容的 `/v1/embeddings` 端点调用本地 Ollama 做真实语义 embedding。
pub(crate) struct OllamaEmbedder {
    client: reqwest::Client,
    endpoint: String,
    api_key: Option<String>,
    model: String,
    dim: usize,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
}

impl OllamaEmbedder {
    /// 构造时向 Ollama 发一个探针请求，确定模型输出维度，避免硬编码。
    pub(crate) async fn new(
        base: &str,
        api_key: Option<String>,
        model: String,
    ) -> Result<Self, BackendError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|error| BackendError::ollama(error.to_string()))?;
        let endpoint = format!("{}/embeddings", base.trim_end_matches('/'));
        let embedder = Self {
            client,
            endpoint: endpoint.clone(),
            api_key,
            model: model.clone(),
            dim: 0,
        };
        let probe = embedder.request(&["dimension probe"]).await?;
        let dim = probe
            .into_iter()
            .next()
            .map(|vector| vector.len())
            .ok_or_else(|| BackendError::ollama("ollama returned no embedding for probe"))?;
        if dim == 0 {
            return Err(BackendError::ollama(
                "ollama returned zero-dimension embedding",
            ));
        }
        Ok(Self { dim, ..embedder })
    }

    async fn request(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>, BackendError> {
        let mut builder = self.client.post(&self.endpoint).json(&serde_json::json!({
            "model": self.model,
            "input": inputs,
        }));
        if let Some(key) = &self.api_key {
            builder = builder.bearer_auth(key);
        }
        let response = builder
            .send()
            .await
            .map_err(|error| BackendError::ollama(error.to_string()))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            eprintln!("ollama embeddings returned {status}: {body}");
            return Err(BackendError::ollama(format!(
                "ollama embeddings returned {status}"
            )));
        }
        let parsed: EmbeddingResponse = response
            .json()
            .await
            .map_err(|error| BackendError::ollama(error.to_string()))?;
        Ok(parsed.data.into_iter().map(|item| item.embedding).collect())
    }
}

#[async_trait]
impl Embedder for OllamaEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, BackendError> {
        let mut results = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(OLLAMA_BATCH) {
            let mut batch = self.request(chunk).await?;
            results.append(&mut batch);
        }
        Ok(results)
    }
}

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector {
            *value /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn hash_embedder_is_deterministic_and_non_zero() {
        let embedder = HashEmbedder::new();
        assert_eq!(embedder.dim(), HASH_DIMENSIONS);
        let a = embedder
            .embed(&["rust makes concurrent systems safer"])
            .await
            .unwrap();
        let b = embedder
            .embed(&["rust makes concurrent systems safer"])
            .await
            .unwrap();
        assert_eq!(a, b);
        assert!(a[0].iter().any(|value| *value != 0.0));
    }

    #[tokio::test]
    async fn hash_embedder_never_returns_all_zero_vector() {
        let embedder = HashEmbedder::new();
        // 仅空白/标点的文本不会产生任何 bigram，原本会得到全零向量。
        let vectors = embedder.embed(&["   "]).await.unwrap();
        let vector = &vectors[0];
        assert!(vector.iter().any(|value| *value != 0.0));
    }

    #[test]
    fn fnv1a_has_reference_values() {
        // FNV-1a 规范测试向量：空串与单字符 "a"。
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
    }

    #[tokio::test]
    async fn hash_embedder_is_stable_across_invocations() {
        // 同一文本在任意进程/版本中都必须得到相同向量，这是 LanceDB 持久化的前提。
        let embedder = HashEmbedder::new();
        let first = embedder.embed(&["跨进程一致的哈希"]).await.unwrap();
        let again = embedder.embed(&["跨进程一致的哈希"]).await.unwrap();
        assert_eq!(first, again);
    }
}
