<p align="center">
  <img src=".github/assets/header.svg" alt="sgrep" width="100%"/>
</p>

<p align="center">
  <code>cargo install --git https://github.com/andrewgazelka/sgrep</code>
</p>

Code search that understands meaning, not just keywords. Combines BM25 lexical search with Jina ColBERT v2 neural embeddings using late interaction for state-of-the-art retrieval.

## Features

- **Hybrid Search**: Fuses BM25 keyword matching with ColBERT semantic similarity via reciprocal rank fusion
- **Late Interaction**: Token-level embeddings with MaxSim scoring preserve nuance that single-vector approaches miss
- **Fast on Apple Silicon**: Native Metal acceleration via Candle (~70ms per inference)
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

Uses [Jina ColBERT v2](https://huggingface.co/jinaai/jina-colbert-v2), an XLM-RoBERTa variant with rotary position embeddings. Model weights download automatically on first run.

## Status

Working CLI with index and search commands. Apple Silicon optimized via Metal GPU acceleration.

---

<details>
<summary>Benchmark (Apple Silicon)</summary>

| Backend | Inference Time |
|---------|----------------|
| Candle + Metal | ~70ms |
| ONNX CPU | ~288ms |
| ONNX CoreML | ~459ms |

</details>
