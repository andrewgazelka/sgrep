//! Embedding generation and ColBERT MaxSim scoring.

use eyre::WrapErr as _;

pub mod maxsim;

/// Compute the ColBERT MaxSim score between a query and document.
///
/// MaxSim computes, for each query token, the maximum similarity to any
/// document token, then sums these maxima.
///
/// Formula: sum over q in Q of max over d in D of sim(q, d)
pub fn maxsim(
    query: &sgrep_core::DocumentEmbedding,
    document: &sgrep_core::DocumentEmbedding,
) -> eyre::Result<f32> {
    // Handle empty cases first (before dimension check)
    if query.embeddings.is_empty() || document.embeddings.is_empty() {
        return Ok(0.0);
    }

    if query.dim != document.dim {
        eyre::bail!(
            "dimension mismatch: query has dim {}, document has dim {}",
            query.dim,
            document.dim
        );
    }

    let score = maxsim::compute_maxsim(&query.embeddings, &document.embeddings);
    Ok(score)
}

/// Batch compute MaxSim scores for a query against multiple documents.
pub fn maxsim_batch(
    query: &sgrep_core::DocumentEmbedding,
    documents: &[sgrep_core::DocumentEmbedding],
) -> eyre::Result<Vec<f32>> {
    documents
        .iter()
        .map(|doc| maxsim(query, doc))
        .collect::<eyre::Result<Vec<_>>>()
        .wrap_err("failed to compute batch maxsim")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(vecs: Vec<Vec<f32>>) -> sgrep_core::DocumentEmbedding {
        let dim = vecs.first().map_or(0, Vec::len);
        sgrep_core::DocumentEmbedding::new(vecs, dim)
    }

    #[test]
    fn test_identical_embeddings_high_score() {
        // Identical embeddings should give high score
        let emb = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let query = make_embedding(emb.clone());
        let doc = make_embedding(emb);

        let score = maxsim(&query, &doc).unwrap();
        // Each query token matches perfectly with itself: dot(q, q) = 1.0
        // So score = 1.0 + 1.0 = 2.0
        assert!((score - 2.0).abs() < 1e-6, "score was {score}");
    }

    #[test]
    fn test_orthogonal_embeddings_zero_score() {
        // Orthogonal embeddings should give zero score
        let query = make_embedding(vec![vec![1.0, 0.0, 0.0]]);
        let doc = make_embedding(vec![vec![0.0, 1.0, 0.0]]);

        let score = maxsim(&query, &doc).unwrap();
        assert!((score - 0.0).abs() < 1e-6, "score was {score}");
    }

    #[test]
    fn test_partial_match() {
        // Query has two tokens, document has one that matches first query token
        let query = make_embedding(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let doc = make_embedding(vec![vec![1.0, 0.0]]);

        let score = maxsim(&query, &doc).unwrap();
        // First query token: max sim with doc = 1.0
        // Second query token: max sim with doc = 0.0
        // Total = 1.0
        assert!((score - 1.0).abs() < 1e-6, "score was {score}");
    }

    #[test]
    fn test_empty_query() {
        let query = make_embedding(vec![]);
        let doc = make_embedding(vec![vec![1.0, 0.0]]);
        let score = maxsim(&query, &doc).unwrap();
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_document() {
        let query = make_embedding(vec![vec![1.0, 0.0]]);
        let doc = make_embedding(vec![]);
        let score = maxsim(&query, &doc).unwrap();
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dimension_mismatch() {
        let query = make_embedding(vec![vec![1.0, 0.0]]);
        let doc = make_embedding(vec![vec![1.0, 0.0, 0.0]]);
        let result = maxsim(&query, &doc);
        assert!(result.is_err());
    }
}
