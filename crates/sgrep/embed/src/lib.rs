//! Embedding generation and ColBERT MaxSim scoring.

use eyre::WrapErr as _;

pub mod gpu;
pub mod maxsim;

pub use gpu::GpuDevice;

/// Compute the ColBERT MaxSim score between a query and document.
///
/// MaxSim computes, for each query token, the maximum similarity to any
/// document token, then sums these maxima.
///
/// Formula: sum over q in Q of max over d in D of sim(q, d)
pub fn maxsim(
    query: sgrep_core::EmbeddingView<'_>,
    document: sgrep_core::EmbeddingView<'_>,
) -> eyre::Result<f32> {
    // Handle empty cases
    if query.nrows() == 0 || document.nrows() == 0 {
        return Ok(0.0);
    }

    // Dimension check
    if query.ncols() != document.ncols() {
        eyre::bail!(
            "dimension mismatch: query has dim {}, document has dim {}",
            query.ncols(),
            document.ncols()
        );
    }

    let score = maxsim::compute_maxsim(query, document);
    Ok(score)
}

/// Batch compute MaxSim scores for a query against multiple documents.
pub fn maxsim_batch(
    query: sgrep_core::EmbeddingView<'_>,
    documents: &[sgrep_core::EmbeddingView<'_>],
) -> eyre::Result<Vec<f32>> {
    documents
        .iter()
        .map(|doc| maxsim(query, *doc))
        .collect::<eyre::Result<Vec<_>>>()
        .wrap_err("failed to compute batch maxsim")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_identical_embeddings_high_score() {
        let emb = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let score = maxsim(emb.view(), emb.view()).unwrap();
        assert!((score - 2.0).abs() < 1e-6, "score was {score}");
    }

    #[test]
    fn test_orthogonal_embeddings_zero_score() {
        let query = array![[1.0, 0.0, 0.0]];
        let doc = array![[0.0, 1.0, 0.0]];
        let score = maxsim(query.view(), doc.view()).unwrap();
        assert!((score - 0.0).abs() < 1e-6, "score was {score}");
    }

    #[test]
    fn test_partial_match() {
        let query = array![[1.0, 0.0], [0.0, 1.0]];
        let doc = array![[1.0, 0.0]];
        let score = maxsim(query.view(), doc.view()).unwrap();
        assert!((score - 1.0).abs() < 1e-6, "score was {score}");
    }

    #[test]
    fn test_empty_query() {
        let query = ndarray::Array2::<f32>::zeros((0, 2));
        let doc = array![[1.0, 0.0]];
        let score = maxsim(query.view(), doc.view()).unwrap();
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_document() {
        let query = array![[1.0, 0.0]];
        let doc = ndarray::Array2::<f32>::zeros((0, 2));
        let score = maxsim(query.view(), doc.view()).unwrap();
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dimension_mismatch() {
        let query = array![[1.0, 0.0]];
        let doc = array![[1.0, 0.0, 0.0]];
        let result = maxsim(query.view(), doc.view());
        assert!(result.is_err());
    }
}
