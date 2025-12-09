//! Content-addressable storage with blake3 hashing.

use std::io::Write as _;

use eyre::WrapErr as _;

/// A content hash using blake3.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContentHash(pub [u8; 32]);

impl ContentHash {
    /// Create a hash from content bytes.
    pub fn from_content(content: &[u8]) -> Self {
        Self(*blake3::hash(content).as_bytes())
    }

    /// Get the hash as a hex string.
    pub fn to_hex(&self) -> String {
        hex::encode(&self.0)
    }

    /// Parse from a hex string.
    pub fn from_hex(s: &str) -> eyre::Result<Self> {
        let bytes = hex::decode(s).map_err(|e| eyre::eyre!("invalid hex string: {e}"))?;
        let arr: [u8; 32] = bytes
            .try_into()
            .map_err(|_| eyre::eyre!("hash must be 32 bytes"))?;
        Ok(Self(arr))
    }
}

impl std::fmt::Display for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// Content-addressable store for embeddings and content.
pub struct CasStore {
    base_path: std::path::PathBuf,
}

impl CasStore {
    /// Create a new CAS store at the given path.
    pub fn new(base_path: std::path::PathBuf) -> eyre::Result<Self> {
        std::fs::create_dir_all(&base_path)
            .wrap_err_with(|| format!("failed to create CAS directory at {base_path:?}"))?;
        Ok(Self { base_path })
    }

    /// Get the path for a given hash.
    fn path_for_hash(&self, hash: &ContentHash, suffix: &str) -> std::path::PathBuf {
        let hex = hash.to_hex();
        // Use first 2 chars as subdirectory for sharding
        let (prefix, rest) = hex.split_at(2);
        self.base_path.join(prefix).join(format!("{rest}{suffix}"))
    }

    /// Store content and return its hash.
    pub fn store_content(&self, content: &[u8]) -> eyre::Result<ContentHash> {
        let hash = ContentHash::from_content(content);
        let path = self.path_for_hash(&hash, "");

        if path.exists() {
            return Ok(hash);
        }

        let parent = path.parent().ok_or_else(|| eyre::eyre!("no parent dir"))?;
        std::fs::create_dir_all(parent)
            .wrap_err_with(|| format!("failed to create directory {parent:?}"))?;

        let file = std::fs::File::create(&path)
            .wrap_err_with(|| format!("failed to create file {path:?}"))?;
        let mut writer = std::io::BufWriter::new(file);
        writer
            .write_all(content)
            .wrap_err("failed to write content")?;

        Ok(hash)
    }

    /// Store embeddings for a content hash.
    pub fn store_embeddings(
        &self,
        hash: &ContentHash,
        embedding: &sgrep_core::DocumentEmbedding,
    ) -> eyre::Result<()> {
        let path = self.path_for_hash(hash, ".emb");

        let parent = path.parent().ok_or_else(|| eyre::eyre!("no parent dir"))?;
        std::fs::create_dir_all(parent)
            .wrap_err_with(|| format!("failed to create directory {parent:?}"))?;

        // Simple binary format: dim (u32), num_tokens (u32), then f32s
        let file = std::fs::File::create(&path)
            .wrap_err_with(|| format!("failed to create embedding file {path:?}"))?;
        let mut writer = std::io::BufWriter::new(file);

        use std::io::Write as _;

        let dim = u32::try_from(embedding.dim).wrap_err("dim too large")?;
        let num_tokens = u32::try_from(embedding.num_tokens()).wrap_err("too many tokens")?;

        writer
            .write_all(&dim.to_le_bytes())
            .wrap_err("failed to write dim")?;
        writer
            .write_all(&num_tokens.to_le_bytes())
            .wrap_err("failed to write num_tokens")?;

        for token_emb in &embedding.embeddings {
            for &val in token_emb {
                writer
                    .write_all(&val.to_le_bytes())
                    .wrap_err("failed to write embedding value")?;
            }
        }

        Ok(())
    }

    /// Load embeddings for a content hash.
    pub fn load_embeddings(
        &self,
        hash: &ContentHash,
    ) -> eyre::Result<Option<sgrep_core::DocumentEmbedding>> {
        use std::io::Read as _;

        let path = self.path_for_hash(hash, ".emb");

        if !path.exists() {
            return Ok(None);
        }

        let file = std::fs::File::open(&path)
            .wrap_err_with(|| format!("failed to open embedding file {path:?}"))?;
        let mut reader = std::io::BufReader::new(file);

        let mut buf4 = [0u8; 4];

        reader
            .read_exact(&mut buf4)
            .wrap_err("failed to read dim")?;
        let dim = u32::from_le_bytes(buf4) as usize;

        reader
            .read_exact(&mut buf4)
            .wrap_err("failed to read num_tokens")?;
        let num_tokens = u32::from_le_bytes(buf4) as usize;

        let mut embeddings = Vec::with_capacity(num_tokens);
        for _ in 0..num_tokens {
            let mut token_emb = Vec::with_capacity(dim);
            for _ in 0..dim {
                reader
                    .read_exact(&mut buf4)
                    .wrap_err("failed to read embedding value")?;
                token_emb.push(f32::from_le_bytes(buf4));
            }
            embeddings.push(token_emb);
        }

        Ok(Some(sgrep_core::DocumentEmbedding::new(embeddings, dim)))
    }

    /// Check if content exists.
    pub fn has_content(&self, hash: &ContentHash) -> bool {
        self.path_for_hash(hash, "").exists()
    }

    /// Check if embeddings exist for a hash.
    pub fn has_embeddings(&self, hash: &ContentHash) -> bool {
        self.path_for_hash(hash, ".emb").exists()
    }
}

// Add hex as a dependency
mod hex {
    const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";

    pub fn encode(bytes: &[u8; 32]) -> String {
        let mut s = String::with_capacity(64);
        for &b in bytes {
            s.push(HEX_CHARS[(b >> 4) as usize] as char);
            s.push(HEX_CHARS[(b & 0xf) as usize] as char);
        }
        s
    }

    pub fn decode(s: &str) -> Result<Vec<u8>, &'static str> {
        if s.len() % 2 != 0 {
            return Err("odd length");
        }

        let mut bytes = Vec::with_capacity(s.len() / 2);
        let chars: Vec<char> = s.chars().collect();

        for chunk in chars.chunks(2) {
            let hi = hex_digit(chunk[0])?;
            let lo = hex_digit(chunk[1])?;
            bytes.push((hi << 4) | lo);
        }

        Ok(bytes)
    }

    fn hex_digit(c: char) -> Result<u8, &'static str> {
        match c {
            '0'..='9' => Ok(c as u8 - b'0'),
            'a'..='f' => Ok(c as u8 - b'a' + 10),
            'A'..='F' => Ok(c as u8 - b'A' + 10),
            _ => Err("invalid hex digit"),
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

        let parsed = ContentHash::from_hex(&hex).unwrap();
        assert_eq!(hash, parsed);
    }

    #[test]
    fn test_cas_store_roundtrip() {
        let tmp = std::env::temp_dir().join("sgrep-cas-test");
        let _ = std::fs::remove_dir_all(&tmp);

        let store = CasStore::new(tmp.clone()).unwrap();

        let content = b"test content";
        let hash = store.store_content(content).unwrap();
        assert!(store.has_content(&hash));

        let embedding =
            sgrep_core::DocumentEmbedding::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], 3);
        store.store_embeddings(&hash, &embedding).unwrap();
        assert!(store.has_embeddings(&hash));

        let loaded = store.load_embeddings(&hash).unwrap().unwrap();
        assert_eq!(loaded.dim, 3);
        assert_eq!(loaded.num_tokens(), 2);

        std::fs::remove_dir_all(&tmp).unwrap();
    }
}
