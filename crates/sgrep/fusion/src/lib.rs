//! Rank fusion combining BM25 and embedding scores.
//!
//! Provides multiple fusion strategies for combining lexical (BM25) and
//! semantic (ColBERT) search results.

use std::collections::HashMap;

/// Reciprocal Rank Fusion (RRF) constant.
/// Higher values give more weight to lower-ranked documents.
const RRF_K: f32 = 60.0;

/// Fuse rankings from multiple sources using Reciprocal Rank Fusion (RRF).
///
/// RRF score = sum over all rankings of 1 / (k + rank)
///
/// This is a simple, effective method that doesn't require score normalization.
pub fn reciprocal_rank_fusion(
    rankings: &[Vec<sgrep_core::SearchResult>],
) -> Vec<sgrep_core::SearchResult> {
    let mut scores: HashMap<String, f32> = HashMap::new();

    for ranking in rankings {
        for (rank, result) in ranking.iter().enumerate() {
            let rrf_score = 1.0 / (RRF_K + (rank as f32) + 1.0);
            *scores.entry(result.doc_id.clone()).or_insert(0.0) += rrf_score;
        }
    }

    let mut fused: Vec<_> = scores
        .into_iter()
        .map(|(doc_id, score)| sgrep_core::SearchResult { doc_id, score })
        .collect();

    // Sort by score descending
    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    fused
}

/// Weighted score fusion with min-max normalization.
///
/// Normalizes each source's scores to [0, 1] using min-max scaling,
/// then combines with specified weights.
///
/// # Arguments
/// * `bm25_results` - BM25 search results
/// * `semantic_results` - Semantic (ColBERT) search results
/// * `bm25_weight` - Weight for BM25 scores (0.0 to 1.0)
///
/// Semantic weight is automatically `1.0 - bm25_weight`.
pub fn weighted_fusion(
    bm25_results: &[sgrep_core::SearchResult],
    semantic_results: &[sgrep_core::SearchResult],
    bm25_weight: f32,
) -> Vec<sgrep_core::SearchResult> {
    let semantic_weight = 1.0 - bm25_weight;

    // Build normalized score maps
    let bm25_normalized = normalize_scores(bm25_results);
    let semantic_normalized = normalize_scores(semantic_results);

    // Combine scores
    let mut combined: HashMap<String, f32> = HashMap::new();

    for (doc_id, score) in &bm25_normalized {
        *combined.entry(doc_id.clone()).or_insert(0.0) += score * bm25_weight;
    }

    for (doc_id, score) in &semantic_normalized {
        *combined.entry(doc_id.clone()).or_insert(0.0) += score * semantic_weight;
    }

    let mut fused: Vec<_> = combined
        .into_iter()
        .map(|(doc_id, score)| sgrep_core::SearchResult { doc_id, score })
        .collect();

    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    fused
}

/// Normalize scores to [0, 1] using min-max scaling.
fn normalize_scores(results: &[sgrep_core::SearchResult]) -> HashMap<String, f32> {
    if results.is_empty() {
        return HashMap::new();
    }

    let max_score = results
        .iter()
        .map(|r| r.score)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_score = results
        .iter()
        .map(|r| r.score)
        .fold(f32::INFINITY, f32::min);

    let range = max_score - min_score;

    results
        .iter()
        .map(|r| {
            let normalized = if range > f32::EPSILON {
                (r.score - min_score) / range
            } else {
                1.0 // All scores equal
            };
            (r.doc_id.clone(), normalized)
        })
        .collect()
}

/// Aggregate chunk scores to file scores.
///
/// When searching returns chunks, this aggregates them back to file-level scores.
/// Uses max score per file (best matching chunk wins).
pub fn aggregate_chunks_to_files(
    chunk_results: &[sgrep_core::SearchResult],
) -> Vec<sgrep_core::SearchResult> {
    let mut file_scores: HashMap<String, f32> = HashMap::new();

    for result in chunk_results {
        // Parse chunk ID to get file path
        let file_path = if let Some((path, _)) = sgrep_chunk::parse_chunk_id(&result.doc_id) {
            path.to_string()
        } else {
            // Not a chunk ID, use as-is
            result.doc_id.clone()
        };

        // Use max score for each file
        let entry = file_scores.entry(file_path).or_insert(f32::NEG_INFINITY);
        if result.score > *entry {
            *entry = result.score;
        }
    }

    let mut files: Vec<_> = file_scores
        .into_iter()
        .map(|(doc_id, score)| sgrep_core::SearchResult { doc_id, score })
        .collect();

    files.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    files
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_results(items: &[(&str, f32)]) -> Vec<sgrep_core::SearchResult> {
        items
            .iter()
            .map(|(id, score)| sgrep_core::SearchResult {
                doc_id: id.to_string(),
                score: *score,
            })
            .collect()
    }

    #[test]
    fn test_rrf_basic() {
        let ranking1 = make_results(&[("a", 10.0), ("b", 5.0), ("c", 1.0)]);
        let ranking2 = make_results(&[("b", 10.0), ("a", 5.0), ("d", 1.0)]);

        let fused = reciprocal_rank_fusion(&[ranking1, ranking2]);

        // Both "a" and "b" appear in both rankings, should be at top
        assert!(fused.len() >= 2);

        // "a" is rank 0 in first (score 1/61) and rank 1 in second (score 1/62)
        // "b" is rank 1 in first (score 1/62) and rank 0 in second (score 1/61)
        // They should have equal scores
        let a_score = fused.iter().find(|r| r.doc_id == "a").unwrap().score;
        let b_score = fused.iter().find(|r| r.doc_id == "b").unwrap().score;
        assert!((a_score - b_score).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_fusion() {
        let bm25 = make_results(&[("a", 10.0), ("b", 5.0)]);
        let semantic = make_results(&[("b", 10.0), ("a", 2.0)]);

        // Equal weights
        let fused = weighted_fusion(&bm25, &semantic, 0.5);

        // Both should have scores
        assert_eq!(fused.len(), 2);

        // "a" has normalized BM25=1.0 (max), semantic=0.0 (min) -> 0.5*1.0 + 0.5*0.0 = 0.5
        // "b" has normalized BM25=0.0 (min), semantic=1.0 (max) -> 0.5*0.0 + 0.5*1.0 = 0.5
        let a_score = fused.iter().find(|r| r.doc_id == "a").unwrap().score;
        let b_score = fused.iter().find(|r| r.doc_id == "b").unwrap().score;
        assert!((a_score - b_score).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_fusion_bm25_bias() {
        let bm25 = make_results(&[("a", 10.0), ("b", 5.0)]);
        let semantic = make_results(&[("b", 10.0), ("a", 2.0)]);

        // Heavy BM25 weight
        let fused = weighted_fusion(&bm25, &semantic, 0.8);

        // "a" should win with BM25 bias
        assert_eq!(fused[0].doc_id, "a");
    }

    #[test]
    fn test_aggregate_chunks() {
        let chunks = make_results(&[
            ("file1.rs#chunk0", 0.8),
            ("file1.rs#chunk1", 0.9), // Best chunk for file1
            ("file2.rs#chunk0", 0.7),
        ]);

        let files = aggregate_chunks_to_files(&chunks);

        assert_eq!(files.len(), 2);
        // file1 should have score 0.9 (max of its chunks)
        let f1 = files.iter().find(|r| r.doc_id == "file1.rs").unwrap();
        assert!((f1.score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_empty_rankings() {
        let fused = reciprocal_rank_fusion(&[]);
        assert!(fused.is_empty());

        let fused = weighted_fusion(&[], &[], 0.5);
        assert!(fused.is_empty());
    }
}
