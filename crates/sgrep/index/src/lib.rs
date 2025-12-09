//! BM25 index using tantivy.

use eyre::WrapErr as _;
use tantivy::schema::Value as _;

/// A BM25 search index for code files.
pub struct Bm25Index {
    index: tantivy::Index,
    schema: tantivy::schema::Schema,
    content_field: tantivy::schema::Field,
    path_field: tantivy::schema::Field,
}

impl Bm25Index {
    /// Create a new in-memory index.
    pub fn new_in_memory() -> eyre::Result<Self> {
        let mut schema_builder = tantivy::schema::Schema::builder();

        // Path field - stored but not indexed for search
        let path_field = schema_builder.add_text_field(
            "path",
            tantivy::schema::TextOptions::default().set_stored(),
        );

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
        let index = tantivy::Index::create_in_ram(schema.clone());

        // Register code tokenizer
        let tokenizer = CodeTokenizer::default();
        index.tokenizers().register("code", tokenizer);

        Ok(Self {
            index,
            schema,
            content_field,
            path_field,
        })
    }

    /// Create a new index at the given path.
    pub fn new_at_path(path: &std::path::Path) -> eyre::Result<Self> {
        std::fs::create_dir_all(path)
            .wrap_err_with(|| format!("failed to create index directory at {path:?}"))?;

        let mut schema_builder = tantivy::schema::Schema::builder();

        let path_field = schema_builder.add_text_field(
            "path",
            tantivy::schema::TextOptions::default().set_stored(),
        );

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
            .wrap_err_with(|| format!("failed to open mmap directory at {path:?}"))?;
        let index = tantivy::Index::open_or_create(dir, schema.clone())
            .wrap_err("failed to open or create index")?;

        let tokenizer = CodeTokenizer::default();
        index.tokenizers().register("code", tokenizer);

        Ok(Self {
            index,
            schema,
            content_field,
            path_field,
        })
    }

    /// Add a document to the index.
    pub fn add_document(&self, path: &str, content: &str) -> eyre::Result<()> {
        let mut writer = self
            .index
            .writer(50_000_000)
            .wrap_err("failed to create index writer")?;

        let mut doc = tantivy::TantivyDocument::default();
        doc.add_text(self.path_field, path);
        doc.add_text(self.content_field, content);

        writer.add_document(doc).wrap_err("failed to add document")?;
        writer.commit().wrap_err("failed to commit")?;

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

/// Custom tokenizer for code that handles camelCase, snake_case, etc.
#[derive(Clone, Default)]
pub struct CodeTokenizer;

impl tantivy::tokenizer::Tokenizer for CodeTokenizer {
    type TokenStream<'a> = CodeTokenStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        CodeTokenStream::new(text)
    }
}

/// Token stream that splits on word boundaries and handles code conventions.
pub struct CodeTokenStream<'a> {
    text: &'a str,
    tokens: Vec<(usize, usize, String)>, // (start, end, lowercase token)
    index: usize,
    token: tantivy::tokenizer::Token,
}

impl<'a> CodeTokenStream<'a> {
    fn new(text: &'a str) -> Self {
        let tokens = tokenize_code(text);
        Self {
            text,
            tokens,
            index: 0,
            token: tantivy::tokenizer::Token::default(),
        }
    }
}

impl tantivy::tokenizer::TokenStream for CodeTokenStream<'_> {
    fn advance(&mut self) -> bool {
        if self.index >= self.tokens.len() {
            return false;
        }

        let (start, end, ref text) = self.tokens[self.index];
        self.token.offset_from = start;
        self.token.offset_to = end;
        self.token.position = self.index;
        self.token.text.clear();
        self.token.text.push_str(text);
        self.index += 1;
        true
    }

    fn token(&self) -> &tantivy::tokenizer::Token {
        &self.token
    }

    fn token_mut(&mut self) -> &mut tantivy::tokenizer::Token {
        &mut self.token
    }
}

/// Tokenize code into words, splitting on camelCase, snake_case, etc.
fn tokenize_code(text: &str) -> Vec<(usize, usize, String)> {
    let mut tokens = Vec::new();
    let mut current_start = 0;
    let mut current_word = String::new();
    let mut _prev_was_upper = false;
    let mut prev_was_lower = false;

    for (i, c) in text.char_indices() {
        let is_word_char = c.is_alphanumeric();
        let is_upper = c.is_uppercase();
        let is_lower = c.is_lowercase();

        if !is_word_char {
            // End of word
            if !current_word.is_empty() {
                tokens.push((current_start, i, current_word.to_lowercase()));
                current_word.clear();
            }
            current_start = i + c.len_utf8();
            _prev_was_upper = false;
            prev_was_lower = false;
            continue;
        }

        // Handle camelCase: split when going from lower to upper
        if is_upper && prev_was_lower && !current_word.is_empty() {
            tokens.push((current_start, i, current_word.to_lowercase()));
            current_word.clear();
            current_start = i;
        }

        // Handle XMLParser -> XML, Parser: split when going from multiple upper to lower
        // e.g., "XMLParser" at 'a' we want to split "XM" from "LParser"... actually "XML" "Parser"
        // This is tricky - we need lookahead or backtrack
        // Simplified: just split on lower->upper for now

        current_word.push(c);
        _prev_was_upper = is_upper;
        prev_was_lower = is_lower;
    }

    // Don't forget the last word
    if !current_word.is_empty() {
        tokens.push((current_start, text.len(), current_word.to_lowercase()));
    }

    // Filter out very short tokens (single chars that aren't meaningful)
    tokens
        .into_iter()
        .filter(|(_, _, t)| t.len() > 1 || t.chars().all(|c| c.is_numeric()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_camel_case() {
        let tokens = tokenize_code("camelCaseExample");
        let words: Vec<_> = tokens.iter().map(|(_, _, t)| t.as_str()).collect();
        assert_eq!(words, vec!["camel", "case", "example"]);
    }

    #[test]
    fn test_tokenize_snake_case() {
        let tokens = tokenize_code("snake_case_example");
        let words: Vec<_> = tokens.iter().map(|(_, _, t)| t.as_str()).collect();
        assert_eq!(words, vec!["snake", "case", "example"]);
    }

    #[test]
    fn test_tokenize_mixed() {
        let tokens = tokenize_code("getUserName_fromDB");
        let words: Vec<_> = tokens.iter().map(|(_, _, t)| t.as_str()).collect();
        assert_eq!(words, vec!["get", "user", "name", "from", "db"]);
    }

    #[test]
    fn test_bm25_basic() {
        let index = Bm25Index::new_in_memory().unwrap();
        index
            .add_document("test.rs", "fn hello_world() { println!(\"hello\"); }")
            .unwrap();
        index
            .add_document("other.rs", "fn goodbye() { println!(\"bye\"); }")
            .unwrap();

        let results = index.search("hello", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, "test.rs");
    }
}
