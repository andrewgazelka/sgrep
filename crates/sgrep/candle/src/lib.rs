//! Candle-based ColBERT inference with Metal GPU acceleration.
//!
//! This crate provides a native Rust implementation of the Jina ColBERT v2 model
//! using the Candle ML framework, with Metal GPU support on Apple Silicon.

mod model;

use eyre::WrapErr as _;

pub use model::JinaColBertConfig;

/// Maximum sequence length for input tokenization.
const MAX_SEQ_LENGTH: usize = 128;

/// A ColBERT encoder using Candle with Metal GPU acceleration.
pub struct ColBertEncoder {
    model: model::JinaColBert,
    tokenizer: tokenizers::Tokenizer,
    device: candle_core::Device,
}

impl ColBertEncoder {
    /// Load a ColBERT encoder from local files.
    ///
    /// # Arguments
    /// * `model_path` - Path to the model.safetensors file
    /// * `tokenizer_path` - Path to the tokenizer.json file
    /// * `config_path` - Path to the config.json file
    pub fn load(
        model_path: impl AsRef<std::path::Path>,
        tokenizer_path: impl AsRef<std::path::Path>,
        config_path: impl AsRef<std::path::Path>,
    ) -> eyre::Result<Self> {
        Self::load_with_device(
            model_path,
            tokenizer_path,
            config_path,
            default_device()?,
        )
    }

    /// Load a ColBERT encoder with a specific device.
    pub fn load_with_device(
        model_path: impl AsRef<std::path::Path>,
        tokenizer_path: impl AsRef<std::path::Path>,
        config_path: impl AsRef<std::path::Path>,
        device: candle_core::Device,
    ) -> eyre::Result<Self> {
        let model_path = model_path.as_ref();
        let tokenizer_path = tokenizer_path.as_ref();
        let config_path = config_path.as_ref();

        tracing::info!(?model_path, ?device, "loading Candle model");

        // Load config
        let config_str = std::fs::read_to_string(config_path)
            .wrap_err_with(|| format!("failed to read config from {config_path:?}"))?;
        let config: JinaColBertConfig = serde_json::from_str(&config_str)
            .wrap_err_with(|| format!("failed to parse config from {config_path:?}"))?;

        tracing::info!(?config, "loaded model config");

        // Load weights
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[model_path],
                candle_core::DType::F32,
                &device,
            )
            .wrap_err_with(|| format!("failed to load weights from {model_path:?}"))?
        };

        // Build model
        let model = model::JinaColBert::new(&config, vb)
            .wrap_err("failed to build model")?;

        // Load tokenizer
        tracing::info!(?tokenizer_path, "loading tokenizer");
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| eyre::eyre!("failed to load tokenizer from {tokenizer_path:?}: {e}"))?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Encode text into a document embedding (per-token embeddings).
    pub fn encode(&mut self, text: &str) -> eyre::Result<sgrep_core::DocumentEmbedding> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| eyre::eyre!("tokenization failed: {e}"))?;

        let token_count = encoding.get_ids().len().min(MAX_SEQ_LENGTH);

        // Prepare input tensors
        let mut input_ids = vec![0u32; MAX_SEQ_LENGTH];
        let mut attention_mask = vec![0u32; MAX_SEQ_LENGTH];

        for (i, &id) in encoding.get_ids().iter().take(MAX_SEQ_LENGTH).enumerate() {
            input_ids[i] = id;
            attention_mask[i] = 1;
        }

        let input_ids = candle_core::Tensor::from_vec(input_ids, (1, MAX_SEQ_LENGTH), &self.device)
            .wrap_err("failed to create input_ids tensor")?;
        let attention_mask =
            candle_core::Tensor::from_vec(attention_mask, (1, MAX_SEQ_LENGTH), &self.device)
                .wrap_err("failed to create attention_mask tensor")?;

        // Run inference
        let output = self
            .model
            .forward(&input_ids, &attention_mask)
            .wrap_err("model forward pass failed")?;

        // Extract embeddings - output shape is [1, seq_len, hidden_dim]
        let output = output
            .to_dtype(candle_core::DType::F32)
            .wrap_err("failed to convert output to f32")?;
        let output = output.to_vec3::<f32>().wrap_err("failed to convert output to vec")?;

        // Only take actual tokens (not padding) and L2 normalize each embedding
        let embeddings: Vec<Vec<f32>> = output
            .into_iter()
            .next()
            .ok_or_else(|| eyre::eyre!("empty output"))?
            .into_iter()
            .take(token_count)
            .map(l2_normalize)
            .collect();

        Ok(sgrep_core::DocumentEmbedding::new(embeddings, model::JinaColBert::OUTPUT_DIM))
    }

    /// Encode multiple texts in a batch.
    pub fn encode_batch(
        &mut self,
        texts: &[&str],
    ) -> eyre::Result<Vec<sgrep_core::DocumentEmbedding>> {
        // For now, encode sequentially. Batch inference can be added later.
        texts.iter().map(|text| self.encode(text)).collect()
    }

    /// Get the device being used.
    #[must_use]
    pub fn device(&self) -> &candle_core::Device {
        &self.device
    }
}

/// Get the default device (Metal on macOS, CPU otherwise).
pub fn default_device() -> eyre::Result<candle_core::Device> {
    #[cfg(feature = "metal")]
    {
        tracing::info!("using Metal device");
        candle_core::Device::new_metal(0).wrap_err("failed to create Metal device")
    }
    #[cfg(all(feature = "cuda", not(feature = "metal")))]
    {
        tracing::info!("using CUDA device");
        candle_core::Device::new_cuda(0).wrap_err("failed to create CUDA device")
    }
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    {
        tracing::info!("using CPU device");
        Ok(candle_core::Device::Cpu)
    }
}

/// Get a CPU device.
#[must_use]
pub fn cpu_device() -> candle_core::Device {
    candle_core::Device::Cpu
}

/// L2 normalize a vector.
fn l2_normalize(v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.into_iter().map(|x| x / norm).collect()
    } else {
        v
    }
}
