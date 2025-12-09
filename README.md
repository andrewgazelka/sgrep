<p align="center">
  <img src=".github/assets/header.svg" alt="sgrep" width="100%"/>
</p>

<p align="center">
  <code>cargo install --git https://github.com/andrewgazelka/sgrep</code>
</p>

Local semantic code search for M-series Macs. No API keys, no cloud, no latency—just fast neural search running entirely on your machine.

## Why sgrep?

**grep finds text. sgrep finds meaning.**

Traditional code search fails when you don't know the exact keywords. Searching "authentication logic" won't find `verify_credentials()`. Searching "error handling" won't find `match result { Err(e) => ... }`.

sgrep uses ColBERT neural embeddings to understand *what code does*, not just what it says. Combined with BM25 for exact matches, you get the best of both worlds.

**Why local?**
- **Private**: Your code never leaves your machine
- **Fast**: ~70ms inference on M-series chips via Metal GPU
- **Offline**: Works without internet after initial model download
- **Free**: No API keys, no cloud costs, no rate limits

## Features

- **Hybrid Search**: Fuses BM25 keyword matching with ColBERT semantic similarity via reciprocal rank fusion
- **Late Interaction**: Token-level embeddings with MaxSim scoring preserve nuance that single-vector approaches miss
- **M-series Optimized**: Native Metal GPU acceleration via Candle
- **Content-Addressed Caching**: Embeddings stored by content hash, survives file renames

## Usage

```bash
# Index a codebase
sgrep index .

# Search semantically
sgrep search "function that handles authentication"

# BM25 only (faster, no ML)
sgrep search --bm25-only "auth"

# JSON output with all scores
sgrep search --json "error handling"
```

## How It Works

1. **Index**: Files are tokenized for BM25 and encoded into 128-dim ColBERT embeddings per token
2. **Search**: Query embeddings are computed, MaxSim finds semantic matches, RRF fuses with BM25 rankings

## Model

Uses [Jina ColBERT v2](https://huggingface.co/jinaai/jina-colbert-v2), an XLM-RoBERTa variant with rotary position embeddings. Model weights download automatically on first run (~500MB).

## Requirements

**macOS with M-series chip** (M1/M2/M3/M4). Built specifically for Apple Silicon using Metal for GPU acceleration.

---

<details>
<summary>Benchmarks</summary>

| Backend | Inference Time |
|---------|----------------|
| Candle + Metal | ~70ms |
| ONNX CPU | ~288ms |
| ONNX CoreML | ~459ms |

</details>
