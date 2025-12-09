//! MaxSim computation for ColBERT-style late interaction.

/// Compute dot product between two vectors.
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute MaxSim score between query and document embeddings.
///
/// For each query token, finds the maximum similarity (dot product) with any
/// document token, then sums all these maxima.
pub fn compute_maxsim(query_embeddings: &[Vec<f32>], doc_embeddings: &[Vec<f32>]) -> f32 {
    query_embeddings
        .iter()
        .map(|q| {
            doc_embeddings
                .iter()
                .map(|d| dot_product(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .filter(|&max_sim| max_sim.is_finite())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        assert!((dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-6);
        assert!((dot_product(&[1.0, 0.0], &[0.0, 1.0]) - 0.0).abs() < 1e-6);
        assert!((dot_product(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_maxsim_basic() {
        let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let doc = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let score = compute_maxsim(&query, &doc);
        assert!((score - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_maxsim_finds_best_match() {
        // Query token should find its best match among document tokens
        let query = vec![vec![1.0, 0.0]];
        let doc = vec![
            vec![0.5, 0.0], // similarity = 0.5
            vec![0.9, 0.0], // similarity = 0.9 (best)
            vec![0.1, 0.0], // similarity = 0.1
        ];
        let score = compute_maxsim(&query, &doc);
        assert!((score - 0.9).abs() < 1e-6);
    }
}
