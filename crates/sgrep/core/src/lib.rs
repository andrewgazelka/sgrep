//! Core types and traits for sgrep.

/// A token embedding - a vector of floats representing a single token.
pub type TokenEmbedding = Vec<f32>;

/// A document's embedding - a matrix where each row is a token embedding.
/// This is the ColBERT representation: one vector per token.
#[derive(Debug, Clone)]
pub struct DocumentEmbedding {
    /// The embeddings for each token in the document.
    /// Shape: [num_tokens, embedding_dim]
    pub embeddings: Vec<TokenEmbedding>,
    /// The dimensionality of each embedding vector.
    pub dim: usize,
}

impl DocumentEmbedding {
    /// Create a new document embedding.
    pub fn new(embeddings: Vec<TokenEmbedding>, dim: usize) -> Self {
        Self { embeddings, dim }
    }

    /// Number of tokens in this document.
    pub fn num_tokens(&self) -> usize {
        self.embeddings.len()
    }
}

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
