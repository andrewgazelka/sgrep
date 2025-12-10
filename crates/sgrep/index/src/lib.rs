//! BM25 index using tantivy.

use eyre::WrapErr as _;
use tantivy::schema::Value as _;

/// A BM25 search index for code files.
pub struct Bm25Index {
    index: tantivy::Index,
    content_field: tantivy::schema::Field,
    path_field: tantivy::schema::Field,
}

/// Build the code tokenizer pipeline:
/// SimpleTokenizer -> CamelCaseSplitter -> LowerCaser -> Stemmer (English) -> RemoveLong
fn build_code_tokenizer() -> tantivy::tokenizer::TextAnalyzer {
    tantivy::tokenizer::TextAnalyzer::builder(tantivy::tokenizer::SimpleTokenizer::default())
        .filter(CamelCaseSplitter)
        .filter(tantivy::tokenizer::LowerCaser)
        .filter(tantivy::tokenizer::Stemmer::new(
            tantivy::tokenizer::Language::English,
        ))
        .filter(tantivy::tokenizer::RemoveLongFilter::limit(40))
        .build()
}

impl Bm25Index {
    /// Create a new in-memory index.
    pub fn new_in_memory() -> eyre::Result<Self> {
        let mut schema_builder = tantivy::schema::Schema::builder();

        // Path field - stored but not indexed for search
        let path_field = schema_builder
            .add_text_field("path", tantivy::schema::TextOptions::default().set_stored());

        // Content field - indexed with custom tokenizer for code
        let text_options = tantivy::schema::TextOptions::default()
            .set_indexing_options(
                tantivy::schema::TextFieldIndexing::default()
                    .set_tokenizer("code")
                    .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions),
            )
            .set_stored();
        let content_field = schema_builder.add_text_field("content", text_options);

        let schema = schema_builder.build();
        let index = tantivy::Index::create_in_ram(schema);

        // Register code tokenizer pipeline
        index.tokenizers().register("code", build_code_tokenizer());

        Ok(Self {
            index,
            content_field,
            path_field,
        })
    }

    /// Create a new index at the given path.
    pub fn new_at_path(path: &std::path::Path) -> eyre::Result<Self> {
        std::fs::create_dir_all(path)
            .wrap_err_with(|| format!("failed to create index directory at {}", path.display()))?;

        let mut schema_builder = tantivy::schema::Schema::builder();

        let path_field = schema_builder
            .add_text_field("path", tantivy::schema::TextOptions::default().set_stored());

        let text_options = tantivy::schema::TextOptions::default()
            .set_indexing_options(
                tantivy::schema::TextFieldIndexing::default()
                    .set_tokenizer("code")
                    .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions),
            )
            .set_stored();
        let content_field = schema_builder.add_text_field("content", text_options);

        let schema = schema_builder.build();
        let dir = tantivy::directory::MmapDirectory::open(path)
            .wrap_err_with(|| format!("failed to open mmap directory at {}", path.display()))?;
        let index = tantivy::Index::open_or_create(dir, schema)
            .wrap_err("failed to open or create index")?;

        index.tokenizers().register("code", build_code_tokenizer());

        Ok(Self {
            index,
            content_field,
            path_field,
        })
    }

    /// Add multiple documents to the index in a single batch.
    /// This is much faster than calling `add_document` repeatedly.
    pub fn add_documents<'a>(
        &self,
        documents: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> eyre::Result<()> {
        let mut writer = self
            .index
            .writer(50_000_000)
            .wrap_err("failed to create index writer")?;

        for (path, content) in documents {
            let mut doc = tantivy::TantivyDocument::default();
            doc.add_text(self.path_field, path);
            doc.add_text(self.content_field, content);

            writer
                .add_document(doc)
                .wrap_err_with(|| format!("failed to add document {path}"))?;
        }

        writer.commit().wrap_err("failed to commit index")?;

        Ok(())
    }

    /// Search the index.
    pub fn search(&self, query: &str, limit: usize) -> eyre::Result<Vec<sgrep_core::SearchResult>> {
        let reader = self
            .index
            .reader()
            .wrap_err("failed to create index reader")?;
        let searcher = reader.searcher();

        let query_parser =
            tantivy::query::QueryParser::for_index(&self.index, vec![self.content_field]);
        let query = query_parser
            .parse_query(query)
            .wrap_err("failed to parse query")?;

        let top_docs = searcher
            .search(&query, &tantivy::collector::TopDocs::with_limit(limit))
            .wrap_err("search failed")?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc: tantivy::TantivyDocument = searcher
                .doc(doc_address)
                .wrap_err("failed to retrieve document")?;

            let path = doc
                .get_first(self.path_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            results.push(sgrep_core::SearchResult {
                doc_id: path,
                score,
            });
        }

        Ok(results)
    }
}

/// Token filter that splits camelCase and PascalCase into separate tokens.
/// e.g., "getUserName" -> "get", "User", "Name"
#[derive(Clone)]
struct CamelCaseSplitter;

impl tantivy::tokenizer::TokenFilter for CamelCaseSplitter {
    type Tokenizer<T: tantivy::tokenizer::Tokenizer> = CamelCaseSplitterFilter<T>;

    fn transform<T: tantivy::tokenizer::Tokenizer>(self, tokenizer: T) -> Self::Tokenizer<T> {
        CamelCaseSplitterFilter { inner: tokenizer }
    }
}

#[derive(Clone)]
struct CamelCaseSplitterFilter<T> {
    inner: T,
}

impl<T: tantivy::tokenizer::Tokenizer> tantivy::tokenizer::Tokenizer for CamelCaseSplitterFilter<T> {
    type TokenStream<'a> = CamelCaseSplitterStream<'a, T::TokenStream<'a>>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        CamelCaseSplitterStream {
            inner: self.inner.token_stream(text),
            pending_tokens: Vec::new(),
            current_token: tantivy::tokenizer::Token::default(),
            position: 0,
            _phantom: std::marker::PhantomData,
        }
    }
}

struct CamelCaseSplitterStream<'a, T> {
    inner: T,
    pending_tokens: Vec<(usize, usize, String)>,
    current_token: tantivy::tokenizer::Token,
    position: usize,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<T: tantivy::tokenizer::TokenStream> tantivy::tokenizer::TokenStream
    for CamelCaseSplitterStream<'_, T>
{
    fn advance(&mut self) -> bool {
        // First, drain any pending tokens from camelCase splitting
        if let Some((offset_from, offset_to, text)) = self.pending_tokens.pop() {
            self.current_token.offset_from = offset_from;
            self.current_token.offset_to = offset_to;
            self.current_token.position = self.position;
            self.current_token.text.clear();
            self.current_token.text.push_str(&text);
            self.position += 1;
            return true;
        }

        // Get next token from inner stream
        if !self.inner.advance() {
            return false;
        }

        let token = self.inner.token();
        let text = &token.text;
        let base_offset = token.offset_from;

        // Split on camelCase boundaries
        let splits = split_camel_case(text);

        if splits.len() <= 1 {
            // No splitting needed, just pass through
            self.current_token.offset_from = token.offset_from;
            self.current_token.offset_to = token.offset_to;
            self.current_token.position = self.position;
            self.current_token.text.clear();
            self.current_token.text.push_str(text);
            self.position += 1;
            return true;
        }

        // Multiple tokens from camelCase split - push all but first to pending (reversed)
        for (start, end, word) in splits.into_iter().rev() {
            self.pending_tokens
                .push((base_offset + start, base_offset + end, word));
        }

        // Pop the first one to return now
        let (offset_from, offset_to, text) = self.pending_tokens.pop().unwrap();
        self.current_token.offset_from = offset_from;
        self.current_token.offset_to = offset_to;
        self.current_token.position = self.position;
        self.current_token.text.clear();
        self.current_token.text.push_str(&text);
        self.position += 1;
        true
    }

    fn token(&self) -> &tantivy::tokenizer::Token {
        &self.current_token
    }

    fn token_mut(&mut self) -> &mut tantivy::tokenizer::Token {
        &mut self.current_token
    }
}

/// Split a word on camelCase boundaries.
/// Returns Vec<(start_offset, end_offset, word)>
fn split_camel_case(text: &str) -> Vec<(usize, usize, String)> {
    let mut tokens = Vec::new();
    let mut current_start = 0;
    let mut current_word = String::new();
    let mut prev_was_lower = false;

    for (i, c) in text.char_indices() {
        let is_upper = c.is_uppercase();
        let is_lower = c.is_lowercase();

        // Split when going from lower to upper (camelCase boundary)
        if is_upper && prev_was_lower && !current_word.is_empty() {
            tokens.push((current_start, i, current_word.clone()));
            current_word.clear();
            current_start = i;
        }

        current_word.push(c);
        prev_was_lower = is_lower;
    }

    // Don't forget the last word
    if !current_word.is_empty() {
        tokens.push((current_start, text.len(), current_word));
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_camel_case() {
        let splits = split_camel_case("camelCaseExample");
        let words: Vec<_> = splits.iter().map(|(_, _, t)| t.as_str()).collect();
        assert_eq!(words, vec!["camel", "Case", "Example"]);
    }

    #[test]
    fn test_split_snake_case() {
        // snake_case is already split by SimpleTokenizer on '_'
        let splits = split_camel_case("snake");
        let words: Vec<_> = splits.iter().map(|(_, _, t)| t.as_str()).collect();
        assert_eq!(words, vec!["snake"]);
    }

    #[test]
    fn test_bm25_basic() {
        let index = Bm25Index::new_in_memory().unwrap();
        index
            .add_documents([
                ("test.rs", "fn hello_world() { println!(\"hello\"); }"),
                ("other.rs", "fn goodbye() { println!(\"bye\"); }"),
            ])
            .unwrap();

        let results = index.search("hello", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, "test.rs");
    }

    #[test]
    fn test_stemming() {
        // Test that stemming works - "running" should match "run"
        let index = Bm25Index::new_in_memory().unwrap();
        index
            .add_documents([
                ("runner.rs", "fn running() {}"),
                ("other.rs", "fn walking() {}"),
            ])
            .unwrap();

        let results = index.search("run", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, "runner.rs");
    }

    #[test]
    fn test_camel_case_search() {
        let index = Bm25Index::new_in_memory().unwrap();
        index
            .add_documents([
                ("user.rs", "fn getUserName() {}"),
                ("other.rs", "fn getAddress() {}"),
            ])
            .unwrap();

        // Should find getUserName when searching for "user"
        let results = index.search("user", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, "user.rs");
    }
}
