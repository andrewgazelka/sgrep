//! Core types and traits for sgrep.

// Link Accelerate BLAS
#[expect(unused_extern_crates, reason = "needed to link BLAS")]
extern crate blas_src;

/// Embedding dimension for ColBERT (Jina ColBERT v2 uses 128).
pub const EMBEDDING_DIM: usize = 128;

/// A document's embedding - a contiguous matrix where each row is a token embedding.
/// Shape: [num_tokens, EMBEDDING_DIM]
///
/// Uses row-major (C) layout for efficient BLAS operations.
pub type Embedding = ndarray::Array2<f32>;

/// A view into an embedding (for zero-copy access from mmap).
pub type EmbeddingView<'a> = ndarray::ArrayView2<'a, f32>;

/// A search result with score and document identifier.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The document identifier (e.g., file path or content hash).
    pub doc_id: String,
    /// The relevance score.
    pub score: f32,
}

/// A ranked list of search results.
pub type SearchResults = Vec<SearchResult>;
