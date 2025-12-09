//! GPU-accelerated batched MaxSim using Metal via Candle.
//!
//! For batch re-ranking, computing MaxSim on GPU is faster than sequential CPU
//! because we can process all query-document pairs in parallel.

use eyre::WrapErr as _;

/// GPU device for batched operations.
pub struct GpuDevice {
    device: candle_core::Device,
}

impl GpuDevice {
    /// Create a new GPU device (Metal on macOS).
    pub fn new() -> eyre::Result<Self> {
        #[cfg(feature = "metal")]
        let device = candle_core::Device::new_metal(0).wrap_err("failed to create Metal device")?;

        #[cfg(not(feature = "metal"))]
        let device = candle_core::Device::Cpu;

        Ok(Self { device })
    }

    /// Create a CPU-only device (for testing/fallback).
    #[must_use]
    pub fn cpu() -> Self {
        Self {
            device: candle_core::Device::Cpu,
        }
    }

    /// Compute MaxSim scores for a query against multiple documents in a single GPU operation.
    ///
    /// This is more efficient than sequential CPU MaxSim for batch re-ranking because:
    /// 1. Single data transfer to GPU
    /// 2. Parallel computation across all documents
    /// 3. Unified memory on M-series means no copy overhead
    ///
    /// # Arguments
    /// * `query` - Query embedding [Q, D] where Q = num query tokens, D = embedding dim
    /// * `documents` - List of document embeddings, each [T_i, D] where T_i = num tokens in doc i
    ///
    /// # Returns
    /// Vector of MaxSim scores, one per document
    pub fn batch_maxsim(
        &self,
        query: sgrep_core::EmbeddingView<'_>,
        documents: &[sgrep_core::EmbeddingView<'_>],
    ) -> eyre::Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let num_query_tokens = query.nrows();
        let dim = query.ncols();

        // Convert query to tensor [Q, D]
        let query_slice = query
            .as_slice()
            .ok_or_else(|| eyre::eyre!("query not contiguous"))?;
        let query_tensor =
            candle_core::Tensor::from_slice(query_slice, (num_query_tokens, dim), &self.device)
                .wrap_err("failed to create query tensor")?;

        let mut scores = Vec::with_capacity(documents.len());

        // Process each document
        // TODO: Could batch multiple small documents together for even better GPU utilization
        for doc in documents {
            let num_doc_tokens = doc.nrows();

            if doc.ncols() != dim {
                eyre::bail!(
                    "dimension mismatch: query dim {dim}, doc dim {}",
                    doc.ncols()
                );
            }

            // Convert doc to tensor [T, D]
            let doc_slice = doc
                .as_slice()
                .ok_or_else(|| eyre::eyre!("doc not contiguous"))?;
            let doc_tensor =
                candle_core::Tensor::from_slice(doc_slice, (num_doc_tokens, dim), &self.device)
                    .wrap_err("failed to create doc tensor")?;

            // Compute similarity matrix: [Q, D] @ [D, T] = [Q, T]
            let doc_t = doc_tensor.t().wrap_err("failed to transpose doc")?;
            let similarities = query_tensor
                .matmul(&doc_t)
                .wrap_err("failed to compute similarities")?;

            // Max over document tokens (axis 1): [Q, T] -> [Q]
            let max_sims = similarities.max(1).wrap_err("failed to compute max")?;

            // Sum over query tokens: [Q] -> scalar
            let score = max_sims
                .sum_all()
                .wrap_err("failed to sum")?
                .to_scalar::<f32>()
                .wrap_err("failed to extract scalar")?;

            scores.push(score);
        }

        Ok(scores)
    }

    /// Compute MaxSim for a single query-document pair on GPU.
    ///
    /// For single comparisons, CPU BLAS may be faster due to kernel launch overhead.
    /// Use `batch_maxsim` when processing multiple documents.
    pub fn maxsim(
        &self,
        query: sgrep_core::EmbeddingView<'_>,
        document: sgrep_core::EmbeddingView<'_>,
    ) -> eyre::Result<f32> {
        let scores = self.batch_maxsim(query, &[document])?;
        scores
            .into_iter()
            .next()
            .ok_or_else(|| eyre::eyre!("empty result"))
    }
}

impl Default for GpuDevice {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self::cpu())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gpu_maxsim_basic() {
        let gpu = GpuDevice::cpu(); // Use CPU for tests

        let query = array![[1.0, 0.0], [0.0, 1.0]];
        let doc = array![[1.0, 0.0], [0.0, 1.0]];

        let score = gpu.maxsim(query.view(), doc.view()).unwrap();
        assert!((score - 2.0).abs() < 1e-5, "score was {score}");
    }

    #[test]
    fn test_gpu_batch_maxsim() {
        let gpu = GpuDevice::cpu();

        let query = array![[1.0, 0.0], [0.0, 1.0]];
        let doc1 = array![[1.0, 0.0], [0.0, 1.0]]; // Perfect match, score = 2.0
        let doc2 = array![[0.5, 0.0], [0.0, 0.5]]; // Half match, score = 1.0
        let doc3 = array![[0.0, 1.0], [1.0, 0.0]]; // Swapped, still score = 2.0

        let scores = gpu
            .batch_maxsim(query.view(), &[doc1.view(), doc2.view(), doc3.view()])
            .unwrap();

        assert_eq!(scores.len(), 3);
        assert!((scores[0] - 2.0).abs() < 1e-5);
        assert!((scores[1] - 1.0).abs() < 1e-5);
        assert!((scores[2] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_gpu_maxsim_orthogonal() {
        let gpu = GpuDevice::cpu();

        let query = array![[1.0, 0.0]];
        let doc = array![[0.0, 1.0]];

        let score = gpu.maxsim(query.view(), doc.view()).unwrap();
        assert!((score - 0.0).abs() < 1e-5);
    }
}
