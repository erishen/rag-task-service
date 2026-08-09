/// 写入向量库 / 关键词索引的一条文本块。向量字段仅在向量库写入时使用，
/// 关键词索引忽略 `vector`，只消费 `text`。
#[derive(Clone)]
pub(crate) struct ChunkRecord {
    pub(crate) document_id: u64,
    pub(crate) chunk_index: usize,
    pub(crate) text: String,
    pub(crate) vector: Vec<f32>,
}

/// 检索命中的一条结果。`score` 语义由产出方决定：
/// 向量库为余弦相似度，BM25 为原始相关性分，RRF 融合后为归一化到 `[0,1]` 的融合分。
#[derive(Clone)]
pub(crate) struct ChunkMatch {
    pub(crate) document_id: u64,
    pub(crate) chunk_index: usize,
    pub(crate) text: String,
    pub(crate) score: f32,
}

/// 检索过滤条件。目前支持按文档范围过滤；`document_ids` 为 `None` 表示全库检索。
/// 向量检索与 BM25 索引共同消费，保证两路召回范围一致（否则 RRF 融合会出错）。
#[derive(Default)]
pub(crate) struct SearchFilter {
    pub(crate) document_ids: Option<std::collections::HashSet<u64>>,
}

impl SearchFilter {
    /// 空范围过滤器：命中为空集，两路检索都应直接返回空结果。
    pub(crate) fn is_empty(&self) -> bool {
        self.document_ids.as_ref().is_some_and(|ids| ids.is_empty())
    }

    /// 判断某文档是否在允许范围内；无过滤时一律放行。
    pub(crate) fn permits(&self, document_id: u64) -> bool {
        self.document_ids
            .as_ref()
            .is_none_or(|ids| ids.contains(&document_id))
    }
}
