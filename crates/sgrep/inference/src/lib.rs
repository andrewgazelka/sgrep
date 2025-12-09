//! ONNX Runtime inference for ColBERT token embeddings.
//!
//! This crate provides a wrapper around ONNX Runtime to run the Jina ColBERT
//! model for generating per-token embeddings. On macOS, it uses the CoreML
//! execution provider for ANE (Apple Neural Engine) acceleration.
//!
//! NOTE: The Candle-based encoder (`sgrep-candle`) is preferred for better
//! Apple Silicon performance via Metal.

use eyre::WrapErr as _;
use ort::execution_providers::ExecutionProvider as _;
use ort::execution_providers::coreml::{
    CoreMLComputeUnits, CoreMLExecutionProvider, CoreMLModelFormat,
};
use ort::session::{Session, builder::GraphOptimizationLevel};

const HIDDEN_DIM: usize = 1024;
const MAX_SEQ_LENGTH: usize = 128;

/// Execution provider configuration for the encoder.
///
/// Benchmarks show CPU-only is fastest for Jina-ColBERT-v2 on Apple Silicon:
/// - CpuOnly: ~200ms (FASTEST)
/// - CoreMLAne: ~275ms (27% slower due to CPU<->ANE data transfer overhead)
/// - CoreMLGpu: FAILS with NeuralNetwork format (unsupported ops)
///
/// MLProgram format (requires macOS 12+/iOS 15+) may have better GPU support.
#[derive(Debug, Clone, Copy, Default)]
pub enum ExecutionProvider {
    /// CPU only - no hardware acceleration. FASTEST for this model.
    #[default]
    CpuOnly,
    /// CoreML with CPU and GPU using NeuralNetwork format. May fail on some models.
    CoreMLGpu,
    /// CoreML with CPU and ANE using NeuralNetwork format. Slower due to partial support.
    CoreMLAne,
    /// CoreML with CPU and GPU using MLProgram format (macOS 12+). Better op support.
    CoreMLGpuMLProgram,
    /// CoreML with CPU and ANE using MLProgram format (macOS 12+). Better op support.
    CoreMLAneMLProgram,
}

/// A ColBERT encoder that uses ONNX Runtime for inference.
pub struct ColBertEncoder {
    session: Session,
    tokenizer: tokenizers::Tokenizer,
}

impl ColBertEncoder {
    /// Load a ColBERT encoder with the default execution provider (CoreML+ANE).
    ///
    /// # Arguments
    /// * `model_path` - Path to the .onnx model file
    /// * `tokenizer_path` - Path to the tokenizer.json file
    pub fn load(
        model_path: impl AsRef<std::path::Path>,
        tokenizer_path: impl AsRef<std::path::Path>,
    ) -> eyre::Result<Self> {
        Self::load_with_provider(model_path, tokenizer_path, ExecutionProvider::default())
    }

    /// Load a ColBERT encoder with a specific execution provider.
    ///
    /// # Arguments
    /// * `model_path` - Path to the .onnx model file
    /// * `tokenizer_path` - Path to the tokenizer.json file
    /// * `provider` - Which execution provider to use
    pub fn load_with_provider(
        model_path: impl AsRef<std::path::Path>,
        tokenizer_path: impl AsRef<std::path::Path>,
        provider: ExecutionProvider,
    ) -> eyre::Result<Self> {
        let model_path = model_path.as_ref();
        let tokenizer_path = tokenizer_path.as_ref();

        tracing::info!(?model_path, ?provider, "loading ONNX model");

        let mut builder = Session::builder()
            .wrap_err("failed to create ONNX session builder")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .wrap_err("failed to set optimization level")?;

        // Register execution provider based on configuration
        match provider {
            ExecutionProvider::CpuOnly => {
                tracing::info!("using CPU-only execution");
            }
            ExecutionProvider::CoreMLGpu => {
                let coreml = CoreMLExecutionProvider::default()
                    .with_compute_units(CoreMLComputeUnits::CPUAndGPU)
                    .with_subgraphs(true);

                if coreml.register(&mut builder).is_ok() {
                    tracing::info!("CoreML execution provider registered (CPU+GPU, NeuralNetwork)");
                } else {
                    tracing::warn!("CoreML not available, falling back to CPU");
                }
            }
            ExecutionProvider::CoreMLAne => {
                let coreml = CoreMLExecutionProvider::default()
                    .with_compute_units(CoreMLComputeUnits::CPUAndNeuralEngine)
                    .with_subgraphs(true);

                if coreml.register(&mut builder).is_ok() {
                    tracing::info!("CoreML execution provider registered (CPU+ANE, NeuralNetwork)");
                } else {
                    tracing::warn!("CoreML not available, falling back to CPU");
                }
            }
            ExecutionProvider::CoreMLGpuMLProgram => {
                let coreml = CoreMLExecutionProvider::default()
                    .with_compute_units(CoreMLComputeUnits::CPUAndGPU)
                    .with_model_format(CoreMLModelFormat::MLProgram)
                    .with_subgraphs(true);

                if coreml.register(&mut builder).is_ok() {
                    tracing::info!("CoreML execution provider registered (CPU+GPU, MLProgram)");
                } else {
                    tracing::warn!("CoreML not available, falling back to CPU");
                }
            }
            ExecutionProvider::CoreMLAneMLProgram => {
                let coreml = CoreMLExecutionProvider::default()
                    .with_compute_units(CoreMLComputeUnits::CPUAndNeuralEngine)
                    .with_model_format(CoreMLModelFormat::MLProgram)
                    .with_subgraphs(true);

                if coreml.register(&mut builder).is_ok() {
                    tracing::info!("CoreML execution provider registered (CPU+ANE, MLProgram)");
                } else {
                    tracing::warn!("CoreML not available, falling back to CPU");
                }
            }
        }

        let session = builder
            .commit_from_file(model_path)
            .wrap_err_with(|| format!("failed to load ONNX model from {}", model_path.display()))?;

        tracing::info!(tokenizer_path = %tokenizer_path.display(), "loading tokenizer");

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| eyre::eyre!("failed to load tokenizer from {}: {e}", tokenizer_path.display()))?;

        Ok(Self { session, tokenizer })
    }

    /// Encode text into a document embedding (per-token embeddings).
    ///
    /// Returns an ndarray with shape [num_tokens, HIDDEN_DIM].
    ///
    /// NOTE: This produces 1024-dim embeddings (raw transformer output).
    /// The Candle encoder produces 128-dim ColBERT embeddings after projection.
    pub fn encode(&mut self, text: &str) -> eyre::Result<ndarray::Array2<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| eyre::eyre!("tokenization failed: {e}"))?;

        let token_count = encoding.get_ids().len().min(MAX_SEQ_LENGTH);

        // Pad or truncate to MAX_SEQ_LENGTH
        let mut input_ids = vec![0i64; MAX_SEQ_LENGTH];
        let mut attention_mask = vec![0i64; MAX_SEQ_LENGTH];

        for (i, &id) in encoding.get_ids().iter().take(MAX_SEQ_LENGTH).enumerate() {
            input_ids[i] = i64::from(id);
            attention_mask[i] = 1;
        }

        // Create input tensors using ort::value::Tensor
        let input_ids_tensor =
            ort::value::Tensor::from_array((vec![1, MAX_SEQ_LENGTH], input_ids.into_boxed_slice()))
                .wrap_err("failed to create input_ids tensor")?;

        let attention_mask_tensor = ort::value::Tensor::from_array((
            vec![1, MAX_SEQ_LENGTH],
            attention_mask.into_boxed_slice(),
        ))
        .wrap_err("failed to create attention_mask tensor")?;

        // Run inference
        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            ])
            .wrap_err("ONNX inference failed")?;

        // Extract output tensor [1, seq_len, hidden_dim]
        let output = outputs
            .get("token_embeddings")
            .ok_or_else(|| eyre::eyre!("model output 'token_embeddings' not found"))?;

        let output_array = output
            .try_extract_array::<f32>()
            .wrap_err("failed to extract output tensor")?;

        // Create ndarray with shape [token_count, HIDDEN_DIM]
        let mut embedding = ndarray::Array2::zeros((token_count, HIDDEN_DIM));
        for i in 0..token_count {
            for j in 0..HIDDEN_DIM {
                embedding[[i, j]] = *output_array.get([0, i, j]).unwrap_or(&0.0);
            }
        }

        Ok(embedding)
    }

    /// Encode multiple texts in a batch (more efficient than encoding one by one).
    pub fn encode_batch(&mut self, texts: &[&str]) -> eyre::Result<Vec<ndarray::Array2<f32>>> {
        // For now, just encode sequentially. Batch inference can be added later.
        texts.iter().map(|text| self.encode(text)).collect()
    }
}

#[cfg(test)]
mod tests {
    // Tests require model files, so they're skipped in CI
    // Run manually with: cargo test --package sgrep-inference -- --ignored

    #[test]
    #[ignore = "requires model files"]
    fn test_encoder_load_and_inference() {
        let mut encoder = super::ColBertEncoder::load(
            "../../scripts/convert/models/jina-colbert-v2.onnx",
            "tokenizer.json",
        )
        .expect("failed to load encoder");

        let embedding = encoder.encode("Hello, world!").expect("encoding failed");

        assert!(!embedding.is_empty());
        assert_eq!(embedding.shape()[1], super::HIDDEN_DIM);
    }
}
