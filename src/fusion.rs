use std::collections::HashMap;

use crate::chunk::ChunkMatch;

/// RRF（Reciprocal Rank Fusion）融合：按 `(document_id, chunk_index)` 去重，
/// 融合分 = Σ 1/(k + rank)，`rank` 从 1 开始计数。同一 chunk 在两列表都出现则累加。
/// 输出按融合分降序，截断 `limit`。`ChunkMatch.text` 优先取向量路（语义更全）。
///
/// 调用方需自行把融合分重缩放到 `[0,1]` 以保持前端百分比语义：
/// `score /= num_lists / (k + 1)`，其中 `num_lists` 是参与融合的列表数。
pub(crate) fn rrf_fuse(
    vector_matches: Vec<ChunkMatch>,
    keyword_matches: Vec<ChunkMatch>,
    limit: usize,
    k: u32,
) -> Vec<ChunkMatch> {
    let mut scores: HashMap<(u64, usize), (f64, ChunkMatch)> = HashMap::new();
    let contribute = |rank: usize| 1.0 / (k as f64 + rank as f64);

    for (i, m) in vector_matches.iter().enumerate() {
        let key = (m.document_id, m.chunk_index);
        let contrib = contribute(i + 1);
        scores
            .entry(key)
            .and_modify(|(score, _)| *score += contrib)
            .or_insert_with(|| (contrib, m.clone()));
    }
    for (i, m) in keyword_matches.iter().enumerate() {
        let key = (m.document_id, m.chunk_index);
        let contrib = contribute(i + 1);
        scores
            .entry(key)
            .and_modify(|(score, _)| *score += contrib)
            .or_insert_with(|| (contrib, m.clone()));
    }

    let mut fused: Vec<(f64, ChunkMatch)> = scores.into_values().collect();
    fused.sort_by(|left, right| right.0.total_cmp(&left.0));
    fused.truncate(limit);
    fused
        .into_iter()
        .map(|(score, mut m)| {
            m.score = score as f32;
            m
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(document_id: u64, chunk_index: usize, score: f32) -> ChunkMatch {
        ChunkMatch {
            document_id,
            chunk_index,
            text: format!("doc {document_id}.{chunk_index}"),
            score,
        }
    }

    #[test]
    fn dedups_and_accumulates_overlap() {
        // 同一 chunk 在两列表都排第 1，融合分应为 2/(k+1)。
        let vector = vec![m(1, 0, 0.9)];
        let keyword = vec![m(1, 0, 5.0)];
        let fused = rrf_fuse(vector, keyword, 10, 60);
        assert_eq!(fused.len(), 1);
        // score 存为 f32，用 f32 容差比较避免精度损失。
        let expected = 2.0_f32 / 61.0_f32;
        assert!((fused[0].score - expected).abs() < 1e-6);
    }

    #[test]
    fn rank_weighting_orders_results() {
        // chunk A：向量第 1、关键词第 1；chunk B：向量第 2、关键词未命中。
        let vector = vec![m(1, 0, 0.9), m(2, 0, 0.8)];
        let keyword = vec![m(1, 0, 5.0)];
        let fused = rrf_fuse(vector, keyword, 10, 60);
        assert_eq!(fused[0].document_id, 1);
        assert_eq!(fused[1].document_id, 2);
    }

    #[test]
    fn empty_keyword_list_returns_vector_truncated() {
        let vector = vec![m(1, 0, 0.9), m(2, 0, 0.8)];
        let fused = rrf_fuse(vector, Vec::new(), 1, 60);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].document_id, 1);
    }

    #[test]
    fn empty_vector_list_returns_keyword_truncated() {
        let keyword = vec![m(3, 0, 5.0)];
        let fused = rrf_fuse(Vec::new(), keyword, 10, 60);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].document_id, 3);
    }
}
