//! Content chunking for semantic search.
//!
//! Splits documents into overlapping chunks for better embedding coverage.

/// A chunk of content with its location in the original document.
#[derive(Debug, Clone)]
pub struct Chunk<'a> {
    /// The chunk content.
    pub content: &'a str,
    /// Byte offset from start of document.
    pub start_byte: usize,
    /// Byte offset of end (exclusive).
    pub end_byte: usize,
    /// Line number where chunk starts (0-indexed).
    pub start_line: usize,
}

/// Configuration for chunking.
#[derive(Debug, Clone, Copy)]
pub struct ChunkConfig {
    /// Target chunk size in characters.
    pub chunk_size: usize,
    /// Overlap between consecutive chunks in characters.
    pub overlap: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1500,  // ~375 tokens at 4 chars/token
            overlap: 200,      // ~50 tokens overlap
        }
    }
}

/// Split text into overlapping chunks, breaking at line boundaries.
///
/// Chunks are sized to fit within embedding model context limits while
/// maintaining enough overlap for continuity.
pub fn chunk_text<'a>(text: &'a str, config: ChunkConfig) -> Vec<Chunk<'a>> {
    if text.is_empty() || text.trim().is_empty() {
        return Vec::new();
    }

    // For small documents, return as single chunk
    if text.len() <= config.chunk_size {
        return vec![Chunk {
            content: text.trim_end(),
            start_byte: 0,
            end_byte: text.len(),
            start_line: 0,
        }];
    }

    let mut chunks = Vec::new();
    let lines: Vec<&str> = text.lines().collect();

    // Build line byte offsets for efficient lookups
    let mut line_offsets: Vec<usize> = Vec::with_capacity(lines.len() + 1);
    let mut offset = 0;
    for line in &lines {
        line_offsets.push(offset);
        offset += line.len() + 1; // +1 for newline
    }
    line_offsets.push(text.len()); // End position

    let mut chunk_start_line = 0;

    while chunk_start_line < lines.len() {
        // Find how many lines we can include in this chunk
        let chunk_start_byte = line_offsets[chunk_start_line];
        let mut chunk_end_line = chunk_start_line;
        let mut chunk_byte_len = 0;

        while chunk_end_line < lines.len() {
            let line_len = lines[chunk_end_line].len() + 1;
            if chunk_byte_len + line_len > config.chunk_size && chunk_end_line > chunk_start_line {
                break;
            }
            chunk_byte_len += line_len;
            chunk_end_line += 1;
        }

        let chunk_end_byte = line_offsets[chunk_end_line].min(text.len());
        let chunk_content = &text[chunk_start_byte..chunk_end_byte];

        if !chunk_content.trim().is_empty() {
            chunks.push(Chunk {
                content: chunk_content.trim_end(),
                start_byte: chunk_start_byte,
                end_byte: chunk_end_byte,
                start_line: chunk_start_line,
            });
        }

        // Calculate next chunk start with overlap
        // Try to go back by overlap amount of bytes
        if chunk_end_line >= lines.len() {
            break; // Done, this was the last chunk
        }

        let mut next_start_line = chunk_end_line;
        let mut overlap_bytes = 0;

        // Walk backwards to include overlap
        while next_start_line > chunk_start_line {
            let prev_line = next_start_line - 1;
            let prev_line_len = lines[prev_line].len() + 1;
            if overlap_bytes + prev_line_len > config.overlap {
                break;
            }
            overlap_bytes += prev_line_len;
            next_start_line = prev_line;
        }

        // Ensure we make progress (at least one line forward from chunk_start)
        if next_start_line <= chunk_start_line {
            next_start_line = chunk_start_line + 1;
        }

        chunk_start_line = next_start_line;
    }

    chunks
}

/// Format a chunk ID from file path and chunk index.
pub fn chunk_id(file_path: &str, chunk_index: usize) -> String {
    format!("{file_path}#chunk{chunk_index}")
}

/// Parse a chunk ID back into file path and chunk index.
pub fn parse_chunk_id(chunk_id: &str) -> Option<(&str, usize)> {
    let (path, suffix) = chunk_id.rsplit_once("#chunk")?;
    let index = suffix.parse().ok()?;
    Some((path, index))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_document_single_chunk() {
        let text = "fn main() {\n    println!(\"hello\");\n}";
        let chunks = chunk_text(text, ChunkConfig::default());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, text);
    }

    #[test]
    fn test_large_document_multiple_chunks() {
        // Create a document larger than chunk size
        // Use shorter lines so overlap can work (20 chars per line)
        let line = "x".repeat(20) + "\n";
        let text = line.repeat(100); // 2100 chars

        let config = ChunkConfig {
            chunk_size: 500,
            overlap: 100,
        };
        let chunks = chunk_text(&text, config);

        assert!(chunks.len() > 1, "should have multiple chunks");

        // Verify chunks cover the whole document
        assert_eq!(chunks[0].start_byte, 0);

        // Verify overlap exists (when lines are short enough for overlap)
        for i in 1..chunks.len() {
            assert!(
                chunks[i].start_byte < chunks[i - 1].end_byte,
                "chunks should overlap: chunk {} starts at {} but chunk {} ends at {}",
                i,
                chunks[i].start_byte,
                i - 1,
                chunks[i - 1].end_byte
            );
        }
    }

    #[test]
    fn test_chunk_id_roundtrip() {
        let id = chunk_id("src/main.rs", 5);
        assert_eq!(id, "src/main.rs#chunk5");

        let (path, idx) = parse_chunk_id(&id).unwrap();
        assert_eq!(path, "src/main.rs");
        assert_eq!(idx, 5);
    }

    #[test]
    fn test_empty_document() {
        let chunks = chunk_text("", ChunkConfig::default());
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_whitespace_only_skipped() {
        let text = "   \n\n   \n";
        let chunks = chunk_text(text, ChunkConfig::default());
        assert!(chunks.is_empty());
    }
}
