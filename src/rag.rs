const CHUNK_SIZE: usize = 800;
const CHUNK_OVERLAP: usize = 120;

/// 按字符窗口切分文本，窗口之间保留一定重叠以避免切断语义。
fn window_chunk(content: &str) -> Vec<String> {
    let characters: Vec<char> = content.chars().collect();
    if characters.is_empty() {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < characters.len() {
        let end = (start + CHUNK_SIZE).min(characters.len());
        let chunk: String = characters[start..end]
            .iter()
            .collect::<String>()
            .trim()
            .to_owned();
        if !chunk.is_empty() {
            chunks.push(chunk);
        }
        if end == characters.len() {
            break;
        }
        start = end.saturating_sub(CHUNK_OVERLAP);
    }
    chunks
}

/// 识别行是否为 ATX Markdown 标题，返回标题级别（1-6）。
fn heading_level(line: &str) -> Option<usize> {
    let trimmed = line.trim();
    let hash_count = trimmed.bytes().take_while(|&b| b == b'#').count();
    if (1..=6).contains(&hash_count) {
        let rest = &trimmed[hash_count..];
        if rest.starts_with(' ') || rest.is_empty() {
            return Some(hash_count);
        }
    }
    None
}

/// 标题感知切块：按 `#` 标题分节，节内再按窗口切分，并把标题祖先链作为前缀
/// 拼进每个 chunk。收益：
/// 1. 不把块切在对标题无信息的空行上，块边界对齐章节；
/// 2. 每个 chunk 自带章节上下文（如 `## LanceDB`），语义检索/BM25 更容易命中
///    "标题即答案"型查询（如搜 "lancedb 查询" 能命中名为《LanceDB》一节的内容）。
///
/// 无标题的纯文本回落为整篇窗口切块，等价于旧的 `chunk_text`。
pub(crate) fn chunk_text(content: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    // 当前节：标题祖先链 + 节正文行。
    let mut ancestors: Vec<String> = Vec::new();
    let mut body: Vec<String> = Vec::new();
    // 保证评测/检索一致性：末尾空行不影响。
    for line in content.lines() {
        match heading_level(line) {
            Some(level) => {
                // 新标题到来：先冲刷上一节。
                if let Some(section) = flatten_section(&ancestors, &body) {
                    chunks.extend(section);
                }
                // 更新祖先链：更低级别标题保留，更高级别标题截断。
                ancestors.truncate(level - 1);
                ancestors.push(line.trim().to_owned());
                body.clear();
            }
            None => body.push(line.to_owned()),
        }
    }
    if let Some(section) = flatten_section(&ancestors, &body) {
        chunks.extend(section);
    }
    if chunks.is_empty() && !body.is_empty() {
        // 纯文本（无标题）整篇窗口切块，兼容旧行为。
        return window_chunk(&body.join("\n"));
    }
    chunks
}

/// 把一个节（标题链 + 正文）切块。标题链作为前缀拼在**每一块**开头，
/// 保证跨多块的章节内容都携带小节标题（检索时按标题/上下文命中）。
fn flatten_section(ancestors: &[String], body: &[String]) -> Option<Vec<String>> {
    let heading = ancestors.join(" ");
    let body_text = body.join("\n");
    if heading.trim().is_empty() && body_text.trim().is_empty() {
        return None;
    }
    // 空正文的节也要产出一条，持有标题本身，否则标题段会丢。
    if body_text.trim().is_empty() {
        return Some(vec![heading]);
    }
    let windowed = window_chunk(&body_text);
    if windowed.is_empty() {
        return None;
    }
    if heading.trim().is_empty() {
        Some(windowed)
    } else {
        Some(
            windowed
                .into_iter()
                .map(|chunk| format!("{heading}\n{chunk}"))
                .collect(),
        )
    }
}

/// 两个已归一化向量的余弦相似度即点积。
pub(crate) fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_short_text_into_one_piece() {
        let chunks = chunk_text("Rust makes concurrent systems safer. ");
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn chunks_overlap_long_text() {
        let long = "a".repeat(CHUNK_SIZE * 2 + 50);
        let chunks = chunk_text(&long);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn headings_anchor_chunk_boundaries() {
        let chunks = chunk_text("# 标题甲\n第一段...\n# 标题乙\n第二段...\n");
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].contains("# 标题甲"));
        assert!(chunks[1].contains("# 标题乙"));
        // 标题不被正文裹挟，独立成块检索。
        assert!(!chunks[0].contains("标题乙"));
        assert!(!chunks[1].contains("标题甲"));
    }

    #[test]
    fn heading_context_is_attached_to_chunks() {
        // 长正文被切成多块时，每块都应带上节标题前缀。
        let body = "内容".repeat(CHUNK_SIZE * 2 + 50);
        let chunks = chunk_text(&format!("# LanceDB\n{body}\n"));
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(
                chunk.starts_with("# LanceDB"),
                "chunk missing heading prefix: {chunk:?}"
            );
        }
    }

    #[test]
    fn empty_heading_only_section_kept() {
        let chunks = chunk_text("# 只有标题\n");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("# 只有标题"));
    }
}
