# sgrep - Semantic Grep

Made for macOS
We designing M-series chips
Semantic code search using ColBERT-style late interaction.

## Model

Uses Jina ColBERT v2 for semantic embeddings.

- **HuggingFace**: https://huggingface.co/jinaai/jina-colbert-v2
- **Paper**: Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever (MRL 2024)
- **Citation**: https://aclanthology.org/2024.mrl-1.11/

### Architecture Notes

Jina ColBERT v2 is an XLM-RoBERTa variant with:
- Rotary Position Embeddings (RoPE)
- Combined Q/K/V projection (`Wqkv`)
- Pre-norm transformer layers (norm before attention/mlp)
- 24 layers, 1024 hidden size, 16 attention heads
- Output projection to 128 dimensions for ColBERT embeddings

Weight naming follows HuggingFace format:
- `roberta.embeddings.word_embeddings.weight`
- `roberta.encoder.layers.N.mixer.Wqkv.{weight,bias}`
- `roberta.encoder.layers.N.mixer.out_proj.{weight,bias}`
- `roberta.encoder.layers.N.mlp.fc1.{weight,bias}`
- `roberta.encoder.layers.N.mlp.fc2.{weight,bias}`
- `roberta.encoder.layers.N.norm1.{weight,bias}` (pre-attention norm)
- `roberta.encoder.layers.N.norm2.{weight,bias}` (pre-mlp norm)
- `roberta.emb_ln.{weight,bias}` (embedding layer norm)
- `linear.weight` (ColBERT projection, no bias)

### Query/Document Encoding

ColBERT uses asymmetric encoding with special marker tokens:
- **Query**: `[CLS] [QueryMarker] text...` (token ID 250002)
- **Document**: `[CLS] [DocumentMarker] text...` (token ID 250003)

Reference implementation: https://github.com/lightonai/pylate

## Inference Backends

### Candle (Recommended)

Native Rust ML framework with Metal GPU support. ~70ms per inference on Apple Silicon.

### ONNX Runtime

Slower on Apple Silicon even with CoreML execution provider (~288ms CPU, ~459ms CoreML).

## Benchmark Results (Apple Silicon)

| Provider | Mean (ms) | Speedup |
|----------|-----------|---------|
| Candle-Metal | 70 | 6.8x |
| ONNX-CPU | 288 | 1.7x |
| ONNX-CoreML | 459 | 1x |
| Candle-CPU | 477 | 1x |
