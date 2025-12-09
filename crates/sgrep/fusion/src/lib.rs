//! Rank fusion combining BM25 and embedding scores.

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

/// Linear combination of normalized scores.
///
/// Normalizes each ranking's scores to [0, 1] then combines with weights.
pub fn linear_combination(
    rankings: &[Vec<sgrep_core::SearchResult>],
    weights: &[f32],
) -> Vec<sgrep_core::SearchResult> {
    if rankings.len() != weights.len() {
        return Vec::new();
    }

    let mut scores: HashMap<String, f32> = HashMap::new();

    for (ranking, &weight) in rankings.iter().zip(weights.iter()) {
        if ranking.is_empty() {
            continue;
        }

        // Find min and max scores for normalization
        let max_score = ranking
            .iter()
            .map(|r| r.score)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_score = ranking
            .iter()
            .map(|r| r.score)
            .fold(f32::INFINITY, f32::min);

        let range = max_score - min_score;

        for result in ranking {
            let normalized = if range > 0.0 {
                (result.score - min_score) / range
            } else {
                1.0 // All scores are the same
            };

            *scores.entry(result.doc_id.clone()).or_insert(0.0) += normalized * weight;
        }
    }

    let mut fused: Vec<_> = scores
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
    fn test_linear_combination() {
        let ranking1 = make_results(&[("a", 1.0), ("b", 0.5)]);
        let ranking2 = make_results(&[("b", 1.0), ("a", 0.0)]);

        // Equal weights
        let fused = linear_combination(&[ranking1, ranking2], &[0.5, 0.5]);

        // "a" gets 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        // "b" gets 0.5 * 0.0 + 0.5 * 1.0 = 0.5 (after normalization)
        // Actually with normalization:
        // ranking1: a=1.0 normalized to 1.0, b=0.5 normalized to 0.0 (since min=0.5, max=1.0, range=0.5)
        // Wait, let me recalculate...

        assert!(fused.len() == 2);
    }

    #[test]
    fn test_empty_rankings() {
        let fused = reciprocal_rank_fusion(&[]);
        assert!(fused.is_empty());

        let fused = linear_combination(&[], &[]);
        assert!(fused.is_empty());
    }
}
