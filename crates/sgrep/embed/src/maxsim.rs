//! MaxSim computation for ColBERT-style late interaction.
//!
//! Uses BLAS matrix multiplication via Accelerate for efficient computation.

use ndarray::Axis;

/// Compute MaxSim score between query and document embeddings using BLAS.
///
/// For each query token, finds the maximum similarity (dot product) with any
/// document token, then sums all these maxima.
///
/// Uses matrix multiplication: `similarities = query @ document.T`
/// Then takes max along axis 1 and sums.
pub fn compute_maxsim(
    query: sgrep_core::EmbeddingView<'_>,
    doc: sgrep_core::EmbeddingView<'_>,
) -> f32 {
    // query: [Q, D], doc: [T, D]
    // similarities: [Q, T] = query @ doc.T
    let similarities = query.dot(&doc.t());

    // For each query token, find max similarity across all doc tokens, then sum
    similarities
        .axis_iter(Axis(0))
        .map(|row| row.iter().copied().fold(f32::NEG_INFINITY, f32::max))
        .filter(|&max_sim| max_sim.is_finite())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_compute_maxsim_basic() {
        let query = array![[1.0, 0.0], [0.0, 1.0]];
        let doc = array![[1.0, 0.0], [0.0, 1.0]];
        let score = compute_maxsim(query.view(), doc.view());
        assert!((score - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_maxsim_finds_best_match() {
        // Query token should find its best match among document tokens
        let query = array![[1.0, 0.0]];
        let doc = array![
            [0.5, 0.0], // similarity = 0.5
            [0.9, 0.0], // similarity = 0.9 (best)
            [0.1, 0.0], // similarity = 0.1
        ];
        let score = compute_maxsim(query.view(), doc.view());
        assert!((score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_orthogonal_zero_score() {
        let query = array![[1.0, 0.0]];
        let doc = array![[0.0, 1.0]];
        let score = compute_maxsim(query.view(), doc.view());
        assert!((score - 0.0).abs() < 1e-6);
    }
}
