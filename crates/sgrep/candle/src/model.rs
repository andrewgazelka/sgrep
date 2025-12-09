//! Jina ColBERT v2 model implementation with Rotary Position Embeddings (RoPE).
//!
//! This is an XLM-RoBERTa variant with RoPE and flash attention style.
//! Weight naming follows the HuggingFace jina-colbert-v2 format.
//!
//! # Architecture
//!
//! - 24 transformer layers with post-norm (norm after residual add)
//! - Combined Q/K/V projection (`Wqkv`) instead of separate projections
//! - Rotary Position Embeddings (RoPE) instead of absolute position embeddings
//! - Output projection to 128 dimensions for ColBERT late interaction
//!
//! # References
//!
//! - Model weights: <https://huggingface.co/jinaai/jina-colbert-v2>
//! - Paper: Jina-ColBERT-v2 (MRL 2024) <https://aclanthology.org/2024.mrl-1.11/>

use candle_core::{Device, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};

/// Configuration for Jina ColBERT v2.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct JinaColBertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f64,
    #[serde(default = "default_rotary_base")]
    pub rotary_emb_base: f64,
}

fn default_rotary_base() -> f64 {
    10000.0
}

impl Default for JinaColBertConfig {
    fn default() -> Self {
        Self {
            vocab_size: 250004,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 8194,
            type_vocab_size: 1,
            layer_norm_eps: 1e-5,
            rotary_emb_base: 10000.0,
        }
    }
}

/// Rotary Position Embedding implementation.
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(dim: usize, max_seq_len: usize, base: f64, device: &Device) -> Result<Self> {
        let half_dim = dim / 2;

        // Compute inverse frequencies
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / (base as f32).powf(i as f32 * 2.0 / dim as f32))
            .collect();

        // Build freqs by computing outer product: positions * inv_freq
        // Instead of matmul, we'll compute this directly to avoid Metal issues
        let mut freqs_data = Vec::with_capacity(max_seq_len * half_dim);
        for pos in 0..max_seq_len {
            for &freq in &inv_freq {
                freqs_data.push(pos as f32 * freq);
            }
        }

        // [max_seq_len, half_dim]
        let freqs = Tensor::from_vec(freqs_data, (max_seq_len, half_dim), device)?;

        // Duplicate for full dim: [max_seq_len, dim]
        let freqs = Tensor::cat(&[&freqs, &freqs], 1)?;

        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        Ok(Self { cos, sin })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, seq_len: usize) -> Result<(Tensor, Tensor)> {
        // q, k shape: [batch, seq, num_heads, head_dim]
        // cos, sin shape after narrow: [seq_len, dim]
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;

        // Reshape for broadcasting to match q/k: [1, seq_len, 1, dim]
        let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

        // Use element-wise operations with explicit broadcasting
        let q_rotated = Self::rotate_half(q)?;
        let k_rotated = Self::rotate_half(k)?;

        // q * cos + rotate_half(q) * sin
        let q_cos = q.broadcast_mul(&cos)?;
        let q_sin = q_rotated.broadcast_mul(&sin)?;
        let q_embed = (q_cos + q_sin)?;

        let k_cos = k.broadcast_mul(&cos)?;
        let k_sin = k_rotated.broadcast_mul(&sin)?;
        let k_embed = (k_cos + k_sin)?;

        Ok((q_embed, k_embed))
    }

    fn rotate_half(x: &Tensor) -> Result<Tensor> {
        let last_dim = x.dims().last().copied().unwrap_or(0);
        let half = last_dim / 2;
        let x1 = x.narrow(candle_core::D::Minus1, 0, half)?;
        let x2 = x.narrow(candle_core::D::Minus1, half, half)?;
        Tensor::cat(&[&x2.neg()?, &x1], candle_core::D::Minus1)
    }
}

/// Layer normalization.
struct LayerNorm {
    inner: candle_nn::LayerNorm,
}

impl LayerNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder<'_>) -> Result<Self> {
        let config = candle_nn::LayerNormConfig {
            eps,
            ..Default::default()
        };
        let inner = candle_nn::layer_norm(size, config, vb)?;
        Ok(Self { inner })
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

/// Self-attention with combined QKV projection and RoPE.
/// Follows jina's "mixer" naming: Wqkv for combined Q/K/V, out_proj for output.
struct SelfAttention {
    wqkv: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    rotary_emb: RotaryEmbedding,
}

impl SelfAttention {
    fn new(config: &JinaColBertConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = hidden_size / num_heads;

        // Combined Q, K, V projection (3 * hidden_size)
        let wqkv = candle_nn::linear(hidden_size, 3 * hidden_size, vb.pp("Wqkv"))?;
        let out_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("out_proj"))?;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.rotary_emb_base,
            vb.device(),
        )?;

        Ok(Self {
            wqkv,
            out_proj,
            num_heads,
            head_dim,
            rotary_emb,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        // Combined Q, K, V projection
        let qkv = self.wqkv.forward(hidden_states)?;

        // Split into Q, K, V - shape [batch, seq, 3 * hidden]
        let hidden_size = self.num_heads * self.head_dim;
        let q = qkv.narrow(2, 0, hidden_size)?;
        let k = qkv.narrow(2, hidden_size, hidden_size)?;
        let v = qkv.narrow(2, 2 * hidden_size, hidden_size)?;

        // Reshape to [batch, seq, num_heads, head_dim]
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;

        // Apply RoPE
        let (q, k) = self.rotary_emb.apply(&q, &k, seq_len)?;

        // Transpose to [batch, num_heads, seq, head_dim]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Compute attention scores
        let scale = (self.head_dim as f64).sqrt();
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let attn_weights = (q.matmul(&k_t)? / scale)?;

        // Apply attention mask if provided
        let attn_weights = if let Some(mask) = attention_mask {
            // mask shape: [batch, seq] -> [batch, 1, 1, seq] for broadcasting
            // attn_weights shape is [batch, heads, seq_q, seq_k]
            let mask = mask.unsqueeze(1)?.unsqueeze(2)?;
            let mask = mask.to_dtype(attn_weights.dtype())?;
            // Convert 0/1 mask to -inf/0 for softmax: where mask=0, add -inf
            let ones = mask.ones_like()?;
            let inverted_mask = (ones - mask)?;
            // Multiply by large negative value to approximate -inf
            const MASK_FILL_VALUE: f64 = -1e9;
            let mask_additive = (inverted_mask * MASK_FILL_VALUE)?;
            attn_weights.broadcast_add(&mask_additive)?
        } else {
            attn_weights
        };

        // Softmax and apply to values
        let attn_weights = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back to [batch, seq, hidden]
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        // Output projection
        self.out_proj.forward(&attn_output)
    }
}

/// Feed-forward network (MLP).
/// Uses fc1/fc2 naming from jina.
struct MLP {
    fc1: Linear,
    fc2: Linear,
}

impl MLP {
    fn new(config: &JinaColBertConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let fc1 = candle_nn::linear(config.hidden_size, config.intermediate_size, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(config.intermediate_size, config.hidden_size, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.gelu()?;
        self.fc2.forward(&x)
    }
}

/// Transformer layer with post-norm (norm after residual add).
///
/// Post-norm architecture:
/// 1. attn_out = mixer(hidden_states)
/// 2. hidden_states = norm1(hidden_states + attn_out)
/// 3. mlp_out = mlp(hidden_states)
/// 4. hidden_states = norm2(hidden_states + mlp_out)
struct TransformerLayer {
    norm1: LayerNorm,
    mixer: SelfAttention,
    norm2: LayerNorm,
    mlp: MLP,
}

impl TransformerLayer {
    fn new(config: &JinaColBertConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let norm1 = LayerNorm::new(config.hidden_size, config.layer_norm_eps, vb.pp("norm1"))?;
        let mixer = SelfAttention::new(config, vb.pp("mixer"))?;
        let norm2 = LayerNorm::new(config.hidden_size, config.layer_norm_eps, vb.pp("norm2"))?;
        let mlp = MLP::new(config, vb.pp("mlp"))?;
        Ok(Self {
            norm1,
            mixer,
            norm2,
            mlp,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Post-norm attention: mixer -> residual add -> norm
        let attn_output = self.mixer.forward(hidden_states, attention_mask)?;
        let hidden_states = self.norm1.forward(&(hidden_states + attn_output)?)?;

        // Post-norm MLP: mlp -> residual add -> norm
        let mlp_output = self.mlp.forward(&hidden_states)?;
        let hidden_states = self.norm2.forward(&(hidden_states + mlp_output)?)?;

        Ok(hidden_states)
    }
}

/// Token embeddings with embedding layer norm.
struct Embeddings {
    word_embeddings: Embedding,
    token_type_embeddings: Embedding,
    emb_ln: LayerNorm,
}

impl Embeddings {
    fn new(config: &JinaColBertConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let word_embeddings = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("embeddings").pp("word_embeddings"),
        )?;
        let token_type_embeddings = candle_nn::embedding(
            config.type_vocab_size,
            config.hidden_size,
            vb.pp("embeddings").pp("token_type_embeddings"),
        )?;
        let emb_ln = LayerNorm::new(config.hidden_size, config.layer_norm_eps, vb.pp("emb_ln"))?;
        Ok(Self {
            word_embeddings,
            token_type_embeddings,
            emb_ln,
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let embeddings = self.word_embeddings.forward(input_ids)?;
        // Add token type embeddings (all zeros for single sequence)
        let token_type_ids = input_ids.zeros_like()?;
        let token_type_embeds = self.token_type_embeddings.forward(&token_type_ids)?;
        let embeddings = (embeddings + token_type_embeds)?;
        self.emb_ln.forward(&embeddings)
    }
}

/// The full Jina ColBERT v2 model.
pub struct JinaColBert {
    embeddings: Embeddings,
    layers: Vec<TransformerLayer>,
    linear: Linear,
}

impl JinaColBert {
    /// Output dimension for ColBERT (128).
    pub const OUTPUT_DIM: usize = 128;

    /// Create a new model from config and weights.
    pub fn new(config: &JinaColBertConfig, vb: VarBuilder<'_>) -> Result<Self> {
        // All weights are under "roberta." prefix
        let roberta_vb = vb.pp("roberta");

        let embeddings = Embeddings::new(config, roberta_vb.clone())?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = TransformerLayer::new(config, roberta_vb.pp("encoder").pp("layers").pp(i))?;
            layers.push(layer);
        }

        // ColBERT projection layer (hidden_size -> OUTPUT_DIM, no bias)
        let linear =
            candle_nn::linear_no_bias(config.hidden_size, Self::OUTPUT_DIM, vb.pp("linear"))?;

        Ok(Self {
            embeddings,
            layers,
            linear,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs, shape [batch, seq_len]
    /// * `attention_mask` - Attention mask, shape [batch, seq_len]
    ///
    /// # Returns
    /// Token embeddings, shape [batch, seq_len, 128]
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.embeddings.forward(input_ids)?;

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, Some(attention_mask))?;
        }

        // Project to ColBERT dimension
        self.linear.forward(&hidden_states)
    }
}
