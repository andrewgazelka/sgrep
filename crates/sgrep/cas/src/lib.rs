//! Embedding storage with two-file format for efficient I/O.
//!
//! Uses two files:
//! - `embeddings.dat`: Append-only data file with concatenated f32 embeddings
//! - `embeddings.idx`: Index file mapping content hashes to offsets
//!
//! This design allows:
//! - Append-only writes to data file (no rewriting)
//! - Single mmap for all reads
//! - Fast batch lookups for GPU operations

use std::io::{Read as _, Seek, Write as _};

use eyre::WrapErr as _;

const DATA_FILE: &str = "embeddings.dat";
const INDEX_FILE: &str = "embeddings.idx";

/// Magic bytes for index file validation.
const INDEX_MAGIC: u32 = 0x5347_5250; // "SGRP"
const INDEX_VERSION: u32 = 1;

/// Size of each index entry: 32 bytes hash + 8 bytes offset + 4 bytes num_tokens = 44 bytes
const INDEX_ENTRY_SIZE: usize = 44;

/// Index file header size: 4 bytes magic + 4 bytes version + 4 bytes count = 12 bytes
const INDEX_HEADER_SIZE: usize = 12;

/// A content hash using blake3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentHash(pub [u8; 32]);

impl ContentHash {
    /// Create a hash from content bytes.
    #[must_use]
    pub fn from_content(content: &[u8]) -> Self {
        Self(*blake3::hash(content).as_bytes())
    }

    /// Get the hash as a hex string.
    #[must_use]
    pub fn to_hex(self) -> String {
        let mut s = String::with_capacity(64);
        for b in self.0 {
            use std::fmt::Write as _;
            let _ = write!(s, "{b:02x}");
        }
        s
    }
}

impl std::fmt::Display for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

/// An entry in the index file.
#[derive(Debug, Clone, Copy)]
struct IndexEntry {
    hash: ContentHash,
    offset: u64,
    num_tokens: u32,
}

impl IndexEntry {
    fn to_bytes(self) -> [u8; INDEX_ENTRY_SIZE] {
        let mut buf = [0u8; INDEX_ENTRY_SIZE];
        buf[..32].copy_from_slice(&self.hash.0);
        buf[32..40].copy_from_slice(&self.offset.to_le_bytes());
        buf[40..44].copy_from_slice(&self.num_tokens.to_le_bytes());
        buf
    }

    fn from_bytes(buf: &[u8; INDEX_ENTRY_SIZE]) -> Self {
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&buf[..32]);

        let offset = u64::from_le_bytes([
            buf[32], buf[33], buf[34], buf[35], buf[36], buf[37], buf[38], buf[39],
        ]);
        let num_tokens = u32::from_le_bytes([buf[40], buf[41], buf[42], buf[43]]);

        Self {
            hash: ContentHash(hash),
            offset,
            num_tokens,
        }
    }
}

/// Memory-mapped embedding store for efficient batch access.
pub struct EmbeddingStore {
    base_path: std::path::PathBuf,
    /// Memory-mapped data file (lazily initialized on first read)
    mmap: Option<memmap2::Mmap>,
    /// In-memory index: hash -> (offset, num_tokens)
    index: std::collections::HashMap<ContentHash, (u64, u32)>,
    /// Buffered file handle for batch writes (avoids repeated open/close)
    write_handle: Option<std::io::BufWriter<std::fs::File>>,
    /// Track if we have unflushed writes
    dirty: bool,
}

impl EmbeddingStore {
    /// Open or create an embedding store at the given path.
    pub fn open(base_path: std::path::PathBuf) -> eyre::Result<Self> {
        std::fs::create_dir_all(&base_path)
            .wrap_err_with(|| format!("failed to create directory {}", base_path.display()))?;

        let mut store = Self {
            base_path,
            mmap: None,
            index: std::collections::HashMap::new(),
            write_handle: None,
            dirty: false,
        };

        // Load existing index if present
        store.load_index()?;

        Ok(store)
    }

    /// Load the index file into memory.
    fn load_index(&mut self) -> eyre::Result<()> {
        let index_path = self.base_path.join(INDEX_FILE);

        if !index_path.exists() {
            return Ok(());
        }

        let mut file = std::fs::File::open(&index_path)
            .wrap_err_with(|| format!("failed to open index file {}", index_path.display()))?;

        // Read header
        let mut header = [0u8; INDEX_HEADER_SIZE];
        file.read_exact(&mut header)
            .wrap_err("failed to read index header")?;

        let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        let count = u32::from_le_bytes([header[8], header[9], header[10], header[11]]);

        if magic != INDEX_MAGIC {
            eyre::bail!("invalid index file magic: expected {INDEX_MAGIC:#x}, got {magic:#x}");
        }

        if version != INDEX_VERSION {
            eyre::bail!("unsupported index version: expected {INDEX_VERSION}, got {version}");
        }

        // Read entries
        self.index.clear();
        self.index.reserve(count as usize);

        let mut entry_buf = [0u8; INDEX_ENTRY_SIZE];
        for _ in 0..count {
            file.read_exact(&mut entry_buf)
                .wrap_err("failed to read index entry")?;
            let entry = IndexEntry::from_bytes(&entry_buf);
            self.index
                .insert(entry.hash, (entry.offset, entry.num_tokens));
        }

        Ok(())
    }

    /// Write the index file from memory.
    fn write_index(&self) -> eyre::Result<()> {
        let index_path = self.base_path.join(INDEX_FILE);

        let file = std::fs::File::create(&index_path)
            .wrap_err_with(|| format!("failed to create index file {}", index_path.display()))?;
        let mut writer = std::io::BufWriter::new(file);

        // Write header
        writer
            .write_all(&INDEX_MAGIC.to_le_bytes())
            .wrap_err("failed to write magic")?;
        writer
            .write_all(&INDEX_VERSION.to_le_bytes())
            .wrap_err("failed to write version")?;

        let count = u32::try_from(self.index.len()).wrap_err("too many entries")?;
        writer
            .write_all(&count.to_le_bytes())
            .wrap_err("failed to write count")?;

        // Write entries
        for (&hash, &(offset, num_tokens)) in &self.index {
            let entry = IndexEntry {
                hash,
                offset,
                num_tokens,
            };
            writer
                .write_all(&entry.to_bytes())
                .wrap_err("failed to write entry")?;
        }

        Ok(())
    }

    /// Ensure the mmap is initialized.
    fn ensure_mmap(&mut self) -> eyre::Result<()> {
        if self.mmap.is_some() {
            return Ok(());
        }

        let data_path = self.base_path.join(DATA_FILE);
        if !data_path.exists() {
            return Ok(());
        }

        let file = std::fs::File::open(&data_path)
            .wrap_err_with(|| format!("failed to open data file {}", data_path.display()))?;

        let mmap = unsafe {
            memmap2::Mmap::map(&file)
                .wrap_err_with(|| format!("failed to mmap {}", data_path.display()))?
        };

        self.mmap = Some(mmap);
        Ok(())
    }

    /// Check if embeddings exist for a hash.
    #[must_use]
    pub fn has_embeddings(&self, hash: &ContentHash) -> bool {
        self.index.contains_key(hash)
    }

    /// Get or create the buffered write handle.
    fn get_write_handle(&mut self) -> eyre::Result<&mut std::io::BufWriter<std::fs::File>> {
        if self.write_handle.is_none() {
            let data_path = self.base_path.join(DATA_FILE);
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&data_path)
                .wrap_err_with(|| format!("failed to open data file {}", data_path.display()))?;
            self.write_handle = Some(std::io::BufWriter::with_capacity(64 * 1024, file));
        }
        Ok(self.write_handle.as_mut().expect("just initialized"))
    }

    /// Store embeddings for a content hash.
    ///
    /// Writes are buffered. Call `flush()` to persist to disk.
    pub fn store_embeddings(
        &mut self,
        hash: &ContentHash,
        embedding: sgrep_core::EmbeddingView<'_>,
    ) -> eyre::Result<()> {
        // Skip if already stored
        if self.index.contains_key(hash) {
            return Ok(());
        }

        let shape = embedding.shape();
        let num_tokens = shape[0];
        let dim = shape[1];

        if dim != sgrep_core::EMBEDDING_DIM {
            eyre::bail!(
                "expected embedding dim {}, got {dim}",
                sgrep_core::EMBEDDING_DIM
            );
        }

        let writer = self.get_write_handle()?;

        // Get current offset (file position)
        let offset = writer
            .stream_position()
            .wrap_err("failed to get stream position")?;

        // Write embedding data
        if embedding.is_standard_layout() {
            let slice = embedding.as_slice().expect("standard layout but no slice");
            let bytes =
                unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), slice.len() * 4) };
            writer
                .write_all(bytes)
                .wrap_err("failed to write embeddings")?;
        } else {
            for &val in embedding.iter() {
                writer
                    .write_all(&val.to_le_bytes())
                    .wrap_err("failed to write embedding value")?;
            }
        }

        // Update in-memory index
        let num_tokens_u32 = u32::try_from(num_tokens).wrap_err("too many tokens")?;
        self.index.insert(*hash, (offset, num_tokens_u32));

        // Mark dirty - index needs to be written on flush
        self.dirty = true;

        // Invalidate mmap (will be recreated on next read after flush)
        self.mmap = None;

        Ok(())
    }

    /// Flush buffered writes to disk and update the index file.
    ///
    /// Call this after a batch of `store_embeddings` calls.
    pub fn flush(&mut self) -> eyre::Result<()> {
        if !self.dirty {
            return Ok(());
        }

        // Flush buffered data
        if let Some(writer) = &mut self.write_handle {
            writer.flush().wrap_err("failed to flush embeddings")?;
        }

        // Write updated index
        self.write_index()?;

        self.dirty = false;
        Ok(())
    }

    /// Get a view into embedding data for a hash.
    ///
    /// Returns `(data_ptr, num_tokens)` for zero-copy access.
    pub fn get_embedding_view(
        &mut self,
        hash: &ContentHash,
    ) -> eyre::Result<Option<sgrep_core::EmbeddingView<'_>>> {
        let Some(&(offset, num_tokens)) = self.index.get(hash) else {
            return Ok(None);
        };

        self.ensure_mmap()?;

        let Some(mmap) = &self.mmap else {
            return Ok(None);
        };

        let num_tokens = num_tokens as usize;
        let byte_offset = offset as usize;
        let num_floats = num_tokens * sgrep_core::EMBEDDING_DIM;
        let byte_len = num_floats * 4;

        // Bounds check
        if byte_offset + byte_len > mmap.len() {
            eyre::bail!(
                "embedding data out of bounds: offset={byte_offset}, len={byte_len}, file_size={}",
                mmap.len()
            );
        }

        // Create view into mmap
        let data_ptr = unsafe { mmap.as_ptr().add(byte_offset) };
        let data_slice = unsafe { std::slice::from_raw_parts(data_ptr.cast::<f32>(), num_floats) };

        let view =
            ndarray::ArrayView2::from_shape((num_tokens, sgrep_core::EMBEDDING_DIM), data_slice)
                .wrap_err("failed to create array view")?;

        Ok(Some(view))
    }

    /// Get multiple embedding views for batch processing.
    ///
    /// Returns views in the same order as input hashes. Missing hashes are skipped.
    pub fn get_batch_views(
        &mut self,
        hashes: &[&ContentHash],
    ) -> eyre::Result<Vec<(ContentHash, sgrep_core::EmbeddingView<'_>)>> {
        self.ensure_mmap()?;

        let Some(mmap) = &self.mmap else {
            return Ok(Vec::new());
        };

        let mut results = Vec::with_capacity(hashes.len());

        for &hash in hashes {
            let Some(&(offset, num_tokens)) = self.index.get(hash) else {
                continue;
            };

            let num_tokens = num_tokens as usize;
            let byte_offset = offset as usize;
            let num_floats = num_tokens * sgrep_core::EMBEDDING_DIM;
            let byte_len = num_floats * 4;

            if byte_offset + byte_len > mmap.len() {
                continue;
            }

            let data_ptr = unsafe { mmap.as_ptr().add(byte_offset) };
            let data_slice =
                unsafe { std::slice::from_raw_parts(data_ptr.cast::<f32>(), num_floats) };

            let view = ndarray::ArrayView2::from_shape(
                (num_tokens, sgrep_core::EMBEDDING_DIM),
                data_slice,
            )
            .wrap_err("failed to create array view")?;

            results.push((*hash, view));
        }

        Ok(results)
    }

    /// Number of stored embeddings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Check if store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Iterate over all stored hashes.
    pub fn hashes(&self) -> impl Iterator<Item = &ContentHash> {
        self.index.keys()
    }
}

impl Drop for EmbeddingStore {
    fn drop(&mut self) {
        // Best-effort flush on drop
        if self.dirty {
            let _ = self.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_hash() {
        let hash = ContentHash::from_content(b"hello world");
        let hex = hash.to_hex();
        assert_eq!(hex.len(), 64);
    }

    #[test]
    fn test_store_roundtrip() {
        let tmp = std::env::temp_dir().join("sgrep-store-test");
        let _ = std::fs::remove_dir_all(&tmp);

        let mut store = EmbeddingStore::open(tmp.clone()).unwrap();

        let content = b"test content";
        let hash = ContentHash::from_content(content);

        // Create embedding
        let embedding = ndarray::Array2::from_shape_fn((3, sgrep_core::EMBEDDING_DIM), |(i, j)| {
            (i * sgrep_core::EMBEDDING_DIM + j) as f32
        });

        // Store and flush
        store.store_embeddings(&hash, embedding.view()).unwrap();
        store.flush().unwrap();
        assert!(store.has_embeddings(&hash));

        // Reload store
        drop(store);
        let mut store = EmbeddingStore::open(tmp.clone()).unwrap();
        assert!(store.has_embeddings(&hash));

        // Load and verify
        let view = store.get_embedding_view(&hash).unwrap().unwrap();
        assert_eq!(view.shape(), &[3, sgrep_core::EMBEDDING_DIM]);

        for i in 0..3 {
            for j in 0..sgrep_core::EMBEDDING_DIM {
                let expected = (i * sgrep_core::EMBEDDING_DIM + j) as f32;
                assert!((view[[i, j]] - expected).abs() < 1e-6);
            }
        }

        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn test_multiple_embeddings() {
        let tmp = std::env::temp_dir().join("sgrep-store-multi-test");
        let _ = std::fs::remove_dir_all(&tmp);

        let mut store = EmbeddingStore::open(tmp.clone()).unwrap();

        // Store multiple embeddings
        for i in 0..5 {
            let content = format!("content {i}");
            let hash = ContentHash::from_content(content.as_bytes());
            let embedding =
                ndarray::Array2::from_shape_fn((i + 2, sgrep_core::EMBEDDING_DIM), |(r, c)| {
                    (i * 1000 + r * 100 + c) as f32
                });
            store.store_embeddings(&hash, embedding.view()).unwrap();
        }
        store.flush().unwrap();

        assert_eq!(store.len(), 5);

        // Verify all can be loaded
        for i in 0..5 {
            let content = format!("content {i}");
            let hash = ContentHash::from_content(content.as_bytes());
            let view = store.get_embedding_view(&hash).unwrap().unwrap();
            assert_eq!(view.shape()[0], i + 2);
        }

        std::fs::remove_dir_all(&tmp).unwrap();
    }
}
