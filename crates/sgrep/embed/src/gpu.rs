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
    /// This batches all documents into a single matrix multiplication for maximum GPU utilization:
    /// 1. Concatenate all document tokens into one [M, D] matrix where M = sum of all doc tokens
    /// 2. Single matmul: [Q, D] @ [D, M] = [Q, M]
    /// 3. Split by document boundaries and compute max per document
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

        // Calculate total tokens and document boundaries
        let mut total_tokens = 0_usize;
        let mut doc_boundaries = Vec::with_capacity(documents.len() + 1);
        doc_boundaries.push(0_usize);

        for doc in documents {
            if doc.ncols() != dim {
                eyre::bail!(
                    "dimension mismatch: query dim {dim}, doc dim {}",
                    doc.ncols()
                );
            }
            total_tokens += doc.nrows();
            doc_boundaries.push(total_tokens);
        }

        // Concatenate all documents into a single [M, D] matrix
        let mut all_docs = Vec::with_capacity(total_tokens * dim);
        for doc in documents {
            let slice = doc
                .as_slice()
                .ok_or_else(|| eyre::eyre!("doc not contiguous"))?;
            all_docs.extend_from_slice(slice);
        }

        // Convert to tensors
        let query_slice = query
            .as_slice()
            .ok_or_else(|| eyre::eyre!("query not contiguous"))?;
        let query_tensor =
            candle_core::Tensor::from_slice(query_slice, (num_query_tokens, dim), &self.device)
                .wrap_err("failed to create query tensor")?;

        let docs_tensor =
            candle_core::Tensor::from_slice(&all_docs, (total_tokens, dim), &self.device)
                .wrap_err("failed to create docs tensor")?;

        // Single batched matmul: [Q, D] @ [D, M] = [Q, M]
        let docs_t = docs_tensor.t().wrap_err("failed to transpose docs")?;
        let similarities = query_tensor
            .matmul(&docs_t)
            .wrap_err("failed to compute similarities")?;

        // Move to CPU for score extraction (small data, faster than GPU splits)
        let similarities = similarities
            .to_device(&candle_core::Device::Cpu)
            .wrap_err("failed to move to CPU")?;
        let sim_data = similarities
            .to_vec2::<f32>()
            .wrap_err("failed to extract similarities")?;

        // Compute MaxSim for each document using boundaries
        let mut scores = Vec::with_capacity(documents.len());
        for doc_idx in 0..documents.len() {
            let start = doc_boundaries[doc_idx];
            let end = doc_boundaries[doc_idx + 1];

            // For each query token, find max sim within this document's range
            let mut total_score = 0.0_f32;
            for query_row in &sim_data {
                let max_sim = query_row[start..end]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                if max_sim.is_finite() {
                    total_score += max_sim;
                }
            }
            scores.push(total_score);
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
