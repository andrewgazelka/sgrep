//! Candle-based ColBERT inference with Metal GPU acceleration.
//!
//! This crate provides a native Rust implementation of the Jina ColBERT v2 model
//! using the Candle ML framework, with Metal GPU support on Apple Silicon.
//!
//! # References
//!
//! - Model: <https://huggingface.co/jinaai/jina-colbert-v2>
//! - Paper: <https://aclanthology.org/2024.mrl-1.11/>
//! - PyLate reference implementation: <https://github.com/lightonai/pylate>

mod model;

use eyre::WrapErr as _;

pub use model::JinaColBertConfig;

/// Maximum sequence length for input tokenization.
const MAX_SEQ_LENGTH: usize = 128;

/// Token ID for [QueryMarker] in jina-colbert-v2 tokenizer.
///
/// ColBERT uses asymmetric encoding with different markers for queries vs documents.
/// See: <https://github.com/lightonai/pylate/blob/main/pylate/models/colbert.py>
const QUERY_MARKER_TOKEN_ID: u32 = 250_002;

/// Token ID for [DocumentMarker] in jina-colbert-v2 tokenizer.
///
/// The marker is inserted after [CLS] token: `[CLS] [Marker] text...`
/// See: <https://github.com/lightonai/pylate/blob/main/pylate/models/colbert.py>
const DOCUMENT_MARKER_TOKEN_ID: u32 = 250_003;

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
        Self::load_with_device(model_path, tokenizer_path, config_path, default_device()?)
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
            .wrap_err_with(|| format!("failed to read config from {}", config_path.display()))?;
        let config: JinaColBertConfig = serde_json::from_str(&config_str)
            .wrap_err_with(|| format!("failed to parse config from {}", config_path.display()))?;

        tracing::info!(?config, "loaded model config");

        // Load weights
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[model_path],
                candle_core::DType::F32,
                &device,
            )
            .wrap_err_with(|| format!("failed to load weights from {}", model_path.display()))?
        };

        // Build model
        let model = model::JinaColBert::new(&config, vb).wrap_err("failed to build model")?;

        // Load tokenizer
        tracing::info!(tokenizer_path = %tokenizer_path.display(), "loading tokenizer");
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| eyre::eyre!("failed to load tokenizer from {}: {e}", tokenizer_path.display()))?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Encode a query into embeddings (uses [QueryMarker] prefix).
    pub fn encode_query(&mut self, text: &str) -> eyre::Result<sgrep_core::Embedding> {
        self.encode_with_marker(text, QUERY_MARKER_TOKEN_ID)
    }

    /// Encode a document into embeddings (uses [DocumentMarker] prefix).
    pub fn encode_document(&mut self, text: &str) -> eyre::Result<sgrep_core::Embedding> {
        self.encode_with_marker(text, DOCUMENT_MARKER_TOKEN_ID)
    }

    /// Encode text with a specific marker token inserted after [CLS].
    fn encode_with_marker(
        &mut self,
        text: &str,
        marker_token_id: u32,
    ) -> eyre::Result<sgrep_core::Embedding> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| eyre::eyre!("tokenization failed: {e}"))?;

        let ids = encoding.get_ids();
        // Token count: CLS + marker + text tokens (up to MAX_SEQ_LENGTH - 2 for CLS and marker)
        let text_token_count = ids.len().saturating_sub(1).min(MAX_SEQ_LENGTH - 2);
        let total_token_count = text_token_count + 2; // +2 for CLS and marker

        // Prepare input tensors: [CLS] [Marker] [text tokens...] [padding...]
        let mut input_ids = vec![0u32; MAX_SEQ_LENGTH];
        let mut attention_mask = vec![0u32; MAX_SEQ_LENGTH];

        // First token is CLS (from tokenizer)
        if !ids.is_empty() {
            input_ids[0] = ids[0]; // CLS token
            attention_mask[0] = 1;
        }

        // Second token is the marker
        input_ids[1] = marker_token_id;
        attention_mask[1] = 1;

        // Rest are text tokens (skip CLS from original)
        for (i, &id) in ids.iter().skip(1).take(MAX_SEQ_LENGTH - 2).enumerate() {
            input_ids[i + 2] = id;
            attention_mask[i + 2] = 1;
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

        // Convert to CPU and extract data
        let output = output
            .to_device(&candle_core::Device::Cpu)
            .wrap_err("failed to move output to CPU")?;

        let output_vec = output
            .to_vec3::<f32>()
            .wrap_err("failed to convert output to vec")?;

        // Take the first batch, only actual tokens (not padding), and build ndarray
        let batch = output_vec
            .into_iter()
            .next()
            .ok_or_else(|| eyre::eyre!("empty output"))?;

        // Create contiguous ndarray and L2 normalize each row
        let mut embedding = ndarray::Array2::zeros((total_token_count, sgrep_core::EMBEDDING_DIM));

        for (i, token_emb) in batch.into_iter().take(total_token_count).enumerate() {
            // L2 normalize
            let norm: f32 = token_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm = if norm > 0.0 { norm } else { 1.0 };

            for (j, val) in token_emb.into_iter().enumerate() {
                embedding[[i, j]] = val / norm;
            }
        }

        Ok(embedding)
    }

    /// Encode multiple queries in a single batched GPU call.
    pub fn encode_queries_batch(&mut self, texts: &[&str]) -> eyre::Result<Vec<sgrep_core::Embedding>> {
        self.encode_batch_with_marker(texts, QUERY_MARKER_TOKEN_ID)
    }

    /// Encode multiple documents in a single batched GPU call.
    ///
    /// This is much faster than calling `encode_document` in a loop because:
    /// 1. Single GPU kernel launch instead of N launches
    /// 2. Better GPU utilization through parallel processing
    /// 3. Amortized memory transfer overhead
    pub fn encode_documents_batch(&mut self, texts: &[&str]) -> eyre::Result<Vec<sgrep_core::Embedding>> {
        self.encode_batch_with_marker(texts, DOCUMENT_MARKER_TOKEN_ID)
    }

    /// Encode a batch of texts with the same marker token.
    fn encode_batch_with_marker(
        &mut self,
        texts: &[&str],
        marker_token_id: u32,
    ) -> eyre::Result<Vec<sgrep_core::Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = texts.len();

        // Pre-allocate contiguous buffers (zeros act as padding)
        let mut all_input_ids = vec![0u32; batch_size * MAX_SEQ_LENGTH];
        let mut all_attention_mask = vec![0u32; batch_size * MAX_SEQ_LENGTH];
        let mut token_counts = Vec::with_capacity(batch_size);

        // Tokenize each text (sequential but writes directly to pre-allocated buffer)
        for (idx, text) in texts.iter().enumerate() {
            let encoding = self
                .tokenizer
                .encode(*text, true)
                .map_err(|e| eyre::eyre!("tokenization failed: {e}"))?;

            let ids = encoding.get_ids();
            let text_token_count = ids.len().saturating_sub(1).min(MAX_SEQ_LENGTH - 2);
            let total_token_count = text_token_count + 2; // +2 for CLS and marker
            token_counts.push(total_token_count);

            let offset = idx * MAX_SEQ_LENGTH;

            // CLS token
            if !ids.is_empty() {
                all_input_ids[offset] = ids[0];
                all_attention_mask[offset] = 1;
            }

            // Marker token
            all_input_ids[offset + 1] = marker_token_id;
            all_attention_mask[offset + 1] = 1;

            // Text tokens
            for (i, &id) in ids.iter().skip(1).take(MAX_SEQ_LENGTH - 2).enumerate() {
                all_input_ids[offset + i + 2] = id;
                all_attention_mask[offset + i + 2] = 1;
            }
        }

        // Create batched tensors [batch_size, seq_len]
        let input_ids = candle_core::Tensor::from_vec(
            all_input_ids,
            (batch_size, MAX_SEQ_LENGTH),
            &self.device,
        )
        .wrap_err("failed to create batched input_ids tensor")?;

        let attention_mask = candle_core::Tensor::from_vec(
            all_attention_mask,
            (batch_size, MAX_SEQ_LENGTH),
            &self.device,
        )
        .wrap_err("failed to create batched attention_mask tensor")?;

        // Single forward pass for entire batch
        let output = self
            .model
            .forward(&input_ids, &attention_mask)
            .wrap_err("batched model forward pass failed")?;

        // Convert to CPU
        let output = output
            .to_dtype(candle_core::DType::F32)
            .wrap_err("failed to convert output to f32")?
            .to_device(&candle_core::Device::Cpu)
            .wrap_err("failed to move output to CPU")?;

        // Shape: [batch_size, seq_len, embedding_dim]
        let output_vec = output
            .to_vec3::<f32>()
            .wrap_err("failed to convert output to vec")?;

        // Extract embeddings for each item in batch
        let mut results = Vec::with_capacity(batch_size);

        for (batch_output, &token_count) in output_vec.into_iter().zip(token_counts.iter()) {
            let mut embedding = ndarray::Array2::zeros((token_count, sgrep_core::EMBEDDING_DIM));

            for (i, token_emb) in batch_output.into_iter().take(token_count).enumerate() {
                // L2 normalize
                let norm: f32 = token_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm = if norm > 0.0 { norm } else { 1.0 };

                for (j, val) in token_emb.into_iter().enumerate() {
                    embedding[[i, j]] = val / norm;
                }
            }

            results.push(embedding);
        }

        Ok(results)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to load encoder from HuggingFace Hub for tests.
    fn load_test_encoder() -> eyre::Result<ColBertEncoder> {
        use hf_hub::api::sync::Api;

        let api = Api::new()?;
        let repo = api.model("jinaai/jina-colbert-v2".to_string());

        let model_path = repo.get("model.safetensors")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let config_path = repo.get("config.json")?;

        ColBertEncoder::load(&model_path, &tokenizer_path, &config_path)
    }

    #[test]
    fn test_single_encode_correctness() -> eyre::Result<()> {
        let mut encoder = load_test_encoder()?;

        let embedding = encoder.encode_document("fn main() { println!(\"hello\"); }")?;

        // Should have reasonable token count (CLS + marker + tokens)
        assert!(embedding.nrows() >= 3, "too few tokens: {}", embedding.nrows());
        assert!(embedding.nrows() <= MAX_SEQ_LENGTH, "too many tokens");

        // Should be normalized (L2 norm ≈ 1 for each row)
        for row in embedding.rows() {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.01,
                "embedding not normalized: norm = {norm}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_batch_encode_matches_single() -> eyre::Result<()> {
        let mut encoder = load_test_encoder()?;

        let texts = &[
            "fn main() {}",
            "let x = 42;",
            "struct Foo { bar: i32 }",
        ];

        // Encode individually
        let single_results: Vec<_> = texts
            .iter()
            .map(|t| encoder.encode_document(t))
            .collect::<eyre::Result<Vec<_>>>()?;

        // Encode as batch
        let batch_results = encoder.encode_documents_batch(texts)?;

        assert_eq!(single_results.len(), batch_results.len());

        // Compare embeddings (should be identical)
        for (single, batch) in single_results.iter().zip(batch_results.iter()) {
            assert_eq!(single.shape(), batch.shape(), "shape mismatch");

            // Check values match within floating point tolerance
            for (s, b) in single.iter().zip(batch.iter()) {
                assert!(
                    (s - b).abs() < 1e-5,
                    "value mismatch: single={s}, batch={b}"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_query_vs_document_markers() -> eyre::Result<()> {
        let mut encoder = load_test_encoder()?;

        let text = "hello world";
        let query_emb = encoder.encode_query(text)?;
        let doc_emb = encoder.encode_document(text)?;

        // Same text should produce same shape
        assert_eq!(query_emb.shape(), doc_emb.shape());

        // But different embeddings (due to different marker tokens)
        let mut any_different = false;
        for (q, d) in query_emb.iter().zip(doc_emb.iter()) {
            if (q - d).abs() > 1e-5 {
                any_different = true;
                break;
            }
        }
        assert!(any_different, "query and document embeddings should differ");

        Ok(())
    }

    /// Compute cosine similarity via MaxSim (ColBERT late interaction).
    fn maxsim(query: &ndarray::Array2<f32>, doc: &ndarray::Array2<f32>) -> f32 {
        let mut total = 0.0_f32;
        for q_row in query.rows() {
            let mut max_sim = f32::NEG_INFINITY;
            for d_row in doc.rows() {
                let sim: f32 = q_row.iter().zip(d_row.iter()).map(|(a, b)| a * b).sum();
                if sim > max_sim {
                    max_sim = sim;
                }
            }
            total += max_sim;
        }
        total / query.nrows() as f32
    }

    /// Semantic quality test: related code should score higher than unrelated.
    #[test]
    fn test_semantic_similarity_ranking() -> eyre::Result<()> {
        let mut encoder = load_test_encoder()?;

        // Query about error handling
        let query = encoder.encode_query("error handling try catch")?;

        // Related: actual error handling code
        let related = encoder.encode_document(
            r#"try {
                doSomething();
            } catch (error) {
                console.error("Failed:", error);
                throw error;
            }"#,
        )?;

        // Unrelated: completely different topic
        let unrelated = encoder.encode_document(
            r#"const colors = ["red", "green", "blue"];
            for (const color of colors) {
                console.log(color);
            }"#,
        )?;

        let related_score = maxsim(&query, &related);
        let unrelated_score = maxsim(&query, &unrelated);

        eprintln!("Query: 'error handling try catch'");
        eprintln!("  Related score: {related_score:.4}");
        eprintln!("  Unrelated score: {unrelated_score:.4}");

        assert!(
            related_score > unrelated_score,
            "related code should score higher: related={related_score:.4} <= unrelated={unrelated_score:.4}"
        );

        // The margin should be meaningful (not just barely higher)
        let margin = related_score - unrelated_score;
        assert!(
            margin > 0.05,
            "margin too small: {margin:.4} (expected > 0.05)"
        );

        Ok(())
    }

    /// Semantic quality test: same concept in different languages should be similar.
    #[test]
    fn test_cross_language_similarity() -> eyre::Result<()> {
        let mut encoder = load_test_encoder()?;

        let query = encoder.encode_query("function that adds two numbers")?;

        let rust_code = encoder.encode_document("fn add(a: i32, b: i32) -> i32 { a + b }")?;
        let python_code = encoder.encode_document("def add(a, b):\n    return a + b")?;
        let js_code = encoder.encode_document("function add(a, b) { return a + b; }")?;

        // Unrelated code
        let unrelated = encoder.encode_document("SELECT * FROM users WHERE active = true")?;

        let rust_score = maxsim(&query, &rust_code);
        let python_score = maxsim(&query, &python_code);
        let js_score = maxsim(&query, &js_code);
        let unrelated_score = maxsim(&query, &unrelated);

        eprintln!("Query: 'function that adds two numbers'");
        eprintln!("  Rust: {rust_score:.4}");
        eprintln!("  Python: {python_score:.4}");
        eprintln!("  JavaScript: {js_score:.4}");
        eprintln!("  SQL (unrelated): {unrelated_score:.4}");

        // All implementations should score higher than unrelated
        assert!(rust_score > unrelated_score, "Rust should beat unrelated");
        assert!(python_score > unrelated_score, "Python should beat unrelated");
        assert!(js_score > unrelated_score, "JS should beat unrelated");

        Ok(())
    }

    /// Semantic quality test: specific function names should match.
    #[test]
    fn test_function_name_matching() -> eyre::Result<()> {
        let mut encoder = load_test_encoder()?;

        let query = encoder.encode_query("parseJSON")?;

        let matching = encoder.encode_document(
            r#"pub fn parse_json(input: &str) -> Result<Value, Error> {
                serde_json::from_str(input)
            }"#,
        )?;

        let similar_topic = encoder.encode_document(
            r#"pub fn to_xml(value: &Value) -> String {
                serialize_to_xml(value)
            }"#,
        )?;

        let unrelated = encoder.encode_document(
            r#"pub fn calculate_distance(a: Point, b: Point) -> f64 {
                ((b.x - a.x).powi(2) + (b.y - a.y).powi(2)).sqrt()
            }"#,
        )?;

        let matching_score = maxsim(&query, &matching);
        let similar_score = maxsim(&query, &similar_topic);
        let unrelated_score = maxsim(&query, &unrelated);

        eprintln!("Query: 'parseJSON'");
        eprintln!("  parse_json function: {matching_score:.4}");
        eprintln!("  to_xml (similar topic): {similar_score:.4}");
        eprintln!("  calculate_distance: {unrelated_score:.4}");

        // Exact match should be best
        assert!(
            matching_score > similar_score,
            "exact match should beat similar topic"
        );
        assert!(
            matching_score > unrelated_score,
            "exact match should beat unrelated"
        );

        Ok(())
    }

    /// Regression test: embeddings should have consistent magnitude.
    #[test]
    fn test_embedding_consistency() -> eyre::Result<()> {
        let mut encoder = load_test_encoder()?;

        // Encode same text multiple times
        let text = "fn main() { println!(\"hello\"); }";
        let emb1 = encoder.encode_document(text)?;
        let emb2 = encoder.encode_document(text)?;

        // Should be identical
        assert_eq!(emb1.shape(), emb2.shape());
        for (a, b) in emb1.iter().zip(emb2.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "embeddings not deterministic: {a} != {b}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_empty_batch() -> eyre::Result<()> {
        let mut encoder = load_test_encoder()?;
        let result = encoder.encode_documents_batch(&[])?;
        assert!(result.is_empty());
        Ok(())
    }

    #[test]
    fn test_long_document_truncation() -> eyre::Result<()> {
        let mut encoder = load_test_encoder()?;

        // Create a very long document
        let long_text = "word ".repeat(1000);
        let embedding = encoder.encode_document(&long_text)?;

        // Should be truncated to MAX_SEQ_LENGTH
        assert!(
            embedding.nrows() <= MAX_SEQ_LENGTH,
            "should truncate: got {} rows",
            embedding.nrows()
        );

        Ok(())
    }
}
