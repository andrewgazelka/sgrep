//! CLI for semantic grep.

// CLI binaries need to print user-facing output
#![allow(
    clippy::print_stdout,
    reason = "CLI binary needs stdout for user output"
)]

use eyre::WrapErr as _;

const MODEL_REPO: &str = "jinaai/jina-colbert-v2";
const BM25_DIR: &str = "bm25";
const EMBEDDINGS_DIR: &str = "embeddings";
const INDEX_DIR: &str = ".sgrep";

fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    match args.command {
        Command::Search {
            query,
            path,
            limit,
            bm25_only,
            json,
        } => {
            search(&query, &path, limit, bm25_only, json)?;
        }
        Command::Index { path } => {
            index(&path)?;
        }
    }

    Ok(())
}

use clap::Parser as _;

#[derive(clap::Parser)]
#[command(name = "sgrep")]
#[command(about = "Semantic grep - combining BM25 and neural embeddings for code search")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Search for code matching a query
    Search {
        /// The search query
        query: String,

        /// Path to search in (defaults to current directory)
        #[arg(short, long, default_value = ".")]
        path: std::path::PathBuf,

        /// Maximum number of results
        #[arg(short, long, default_value = "10")]
        limit: usize,

        /// Only use BM25 (skip neural embeddings)
        #[arg(long)]
        bm25_only: bool,

        /// Output detailed JSON with BM25, ColBERT, and fusion scores
        #[arg(long)]
        json: bool,
    },

    /// Index a directory for searching
    Index {
        /// Path to index
        path: std::path::PathBuf,
    },
}

/// Load or download the ColBERT model from HuggingFace.
fn load_encoder() -> eyre::Result<sgrep_candle::ColBertEncoder> {
    use hf_hub::api::sync::Api;

    eprintln!("Loading ColBERT model from {MODEL_REPO}...");

    let api = Api::new().wrap_err("failed to create HuggingFace API client")?;
    let repo = api.model(MODEL_REPO.to_string());

    let model_path = repo
        .get("model.safetensors")
        .wrap_err("failed to download model.safetensors")?;
    let tokenizer_path = repo
        .get("tokenizer.json")
        .wrap_err("failed to download tokenizer.json")?;
    let config_path = repo
        .get("config.json")
        .wrap_err("failed to download config.json")?;

    sgrep_candle::ColBertEncoder::load(&model_path, &tokenizer_path, &config_path)
        .wrap_err("failed to load ColBERT encoder")
}

/// Create a file walker that respects gitignore and ignores internal directories.
fn create_walker(path: &std::path::Path) -> eyre::Result<ignore::Walk> {
    use eyre::WrapErr as _;

    let mut builder = ignore::WalkBuilder::new(path);
    builder
        .hidden(false)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true);

    // Always ignore .sgrep and .git directories
    let mut overrides = ignore::overrides::OverrideBuilder::new(path);
    overrides
        .add("!.sgrep/")
        .wrap_err("failed to add .sgrep override")?;
    overrides
        .add("!.git/")
        .wrap_err("failed to add .git override")?;
    builder.overrides(overrides.build().wrap_err("failed to build overrides")?);

    Ok(builder.build())
}

/// Build a mapping from chunk_id to content hash for semantic search.
/// This walks all files and chunks them to find which chunk IDs have embeddings.
fn build_chunk_hash_map(
    store: &sgrep_cas::EmbeddingStore,
    path: &std::path::Path,
) -> eyre::Result<std::collections::HashMap<String, sgrep_cas::ContentHash>> {
    let mut map = std::collections::HashMap::new();
    let chunk_config = sgrep_chunk::ChunkConfig::default();

    let walker = create_walker(path)?;

    for entry in walker {
        let entry = entry.wrap_err("failed to read directory entry")?;
        let file_path = entry.path();

        if !file_path.is_file() {
            continue;
        }

        let content = match std::fs::read(file_path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let content_str = match std::str::from_utf8(&content) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let relative_path = file_path
            .strip_prefix(path)
            .unwrap_or(file_path)
            .to_string_lossy()
            .to_string();

        let chunks = sgrep_chunk::chunk_text(content_str, chunk_config);

        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let chunk_id = sgrep_chunk::chunk_id(&relative_path, chunk_idx);
            let formatted = format_for_indexing(&chunk_id, chunk.content);
            let hash = sgrep_cas::ContentHash::from_content(formatted.as_bytes());

            if store.has_embeddings(&hash) {
                map.insert(chunk_id, hash);
            }
        }
    }

    Ok(map)
}

/// JSON output for detailed search results.
#[derive(serde::Serialize)]
struct JsonResult {
    path: String,
    bm25_score: Option<f32>,
    colbert_score: Option<f32>,
    fusion_score: f32,
}

fn search(
    query: &str,
    path: &std::path::Path,
    limit: usize,
    bm25_only: bool,
    json: bool,
) -> eyre::Result<()> {
    let index_path = path.join(INDEX_DIR);
    if !index_path.exists() {
        eyre::bail!(
            "no index found at {}, run `sgrep index {}` first",
            index_path.display(),
            path.display()
        );
    }

    // BM25 search - results are now chunk IDs
    let bm25_path = index_path.join(BM25_DIR);
    let bm25_index =
        sgrep_index::Bm25Index::new_at_path(&bm25_path).wrap_err("failed to open BM25 index")?;

    // Get more chunk results for fusion (we'll aggregate to files and re-rank)
    let fetch_limit = limit * 10; // More chunks since we aggregate to files
    let bm25_chunk_results = bm25_index
        .search(query, fetch_limit)
        .wrap_err("BM25 search failed")?;

    // Aggregate BM25 chunk results to file-level scores
    let bm25_file_results = sgrep_fusion::aggregate_chunks_to_files(&bm25_chunk_results);

    // Build file-level BM25 score map
    let bm25_scores: std::collections::HashMap<String, f32> = bm25_file_results
        .iter()
        .map(|r| (r.doc_id.clone(), r.score))
        .collect();

    if bm25_only || bm25_file_results.is_empty() {
        if json {
            let results: Vec<JsonResult> = bm25_file_results
                .iter()
                .take(limit)
                .map(|r| JsonResult {
                    path: r.doc_id.clone(),
                    bm25_score: Some(r.score),
                    colbert_score: None,
                    fusion_score: r.score,
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&results).wrap_err("failed to serialize JSON")?
            );
        } else {
            for result in bm25_file_results.iter().take(limit) {
                println!("{}: {:.4}", result.doc_id, result.score);
            }
        }
        return Ok(());
    }

    // Semantic search with ColBERT
    let embeddings_path = index_path.join(EMBEDDINGS_DIR);
    let mut store = sgrep_cas::EmbeddingStore::open(embeddings_path)
        .wrap_err("failed to open embedding store")?;

    // Build chunk_id -> hash mapping
    let chunk_hash_map =
        build_chunk_hash_map(&store, path).wrap_err("failed to build chunk hash map")?;

    if chunk_hash_map.is_empty() {
        // No embeddings, fall back to BM25 only
        eprintln!("Warning: no embeddings found, using BM25 only");
        if json {
            let results: Vec<JsonResult> = bm25_file_results
                .iter()
                .take(limit)
                .map(|r| JsonResult {
                    path: r.doc_id.clone(),
                    bm25_score: Some(r.score),
                    colbert_score: None,
                    fusion_score: r.score,
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&results).wrap_err("failed to serialize JSON")?
            );
        } else {
            for result in bm25_file_results.iter().take(limit) {
                println!("{}: {:.4}", result.doc_id, result.score);
            }
        }
        return Ok(());
    }

    // Load encoder and encode query
    let mut encoder = load_encoder()?;
    let query_embedding = encoder
        .encode_query(query)
        .wrap_err("failed to encode query")?;

    // Collect chunk embeddings for batch processing
    // Use the chunks from BM25 results that have embeddings
    let mut chunk_ids = Vec::new();
    let mut chunk_hashes = Vec::new();

    for bm25_result in &bm25_chunk_results {
        if let Some(hash) = chunk_hash_map.get(&bm25_result.doc_id) {
            chunk_ids.push(bm25_result.doc_id.clone());
            chunk_hashes.push(hash);
        }
    }

    // Get batch views and compute MaxSim scores on GPU
    let gpu = sgrep_embed::GpuDevice::new().wrap_err("failed to create GPU device")?;

    let batch_views = store
        .get_batch_views(&chunk_hashes)
        .wrap_err("failed to get batch views")?;

    // Extract just the views for batch processing
    let chunk_views: Vec<_> = batch_views.iter().map(|(_, v)| *v).collect();

    let colbert_scores_vec = gpu
        .batch_maxsim(query_embedding.view(), &chunk_views)
        .wrap_err("failed to compute batch MaxSim")?;

    // Build chunk-level semantic results
    let mut semantic_chunk_results = Vec::new();

    for (i, (hash, _)) in batch_views.iter().enumerate() {
        // Find chunk_id for this hash
        let chunk_id = chunk_hashes
            .iter()
            .zip(chunk_ids.iter())
            .find(|(h, _)| **h == hash)
            .map(|(_, id)| id.clone());

        if let Some(chunk_id) = chunk_id {
            let score = colbert_scores_vec[i];
            semantic_chunk_results.push(sgrep_core::SearchResult {
                doc_id: chunk_id,
                score,
            });
        }
    }

    // Aggregate semantic chunk results to file-level scores
    let semantic_file_results = sgrep_fusion::aggregate_chunks_to_files(&semantic_chunk_results);

    // Build file-level ColBERT score map
    let colbert_scores: std::collections::HashMap<String, f32> = semantic_file_results
        .iter()
        .map(|r| (r.doc_id.clone(), r.score))
        .collect();

    // Fuse file-level results using weighted fusion (more mathematically sound than RRF)
    // BM25 weight of 0.3 gives slight preference to semantic search for code
    let fused = sgrep_fusion::weighted_fusion(&bm25_file_results, &semantic_file_results, 0.3);

    if json {
        let results: Vec<JsonResult> = fused
            .iter()
            .take(limit)
            .map(|r| JsonResult {
                path: r.doc_id.clone(),
                bm25_score: bm25_scores.get(&r.doc_id).copied(),
                colbert_score: colbert_scores.get(&r.doc_id).copied(),
                fusion_score: r.score,
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&results).wrap_err("failed to serialize JSON")?
        );
    } else {
        for result in fused.iter().take(limit) {
            println!("{}: {:.4}", result.doc_id, result.score);
        }
    }

    Ok(())
}

/// File to be indexed with its content pre-loaded.
struct IndexableFile {
    relative_path: String,
    content_bytes: Vec<u8>,
}

/// A chunk of a file ready for indexing.
struct IndexableChunk {
    /// Chunk ID in format "file_path#chunkN"
    chunk_id: String,
    /// The file this chunk belongs to
    file_path: String,
    /// Formatted content with path header
    formatted_content: String,
    /// Content hash for the chunk
    hash: sgrep_cas::ContentHash,
}

/// Format content with file path header for indexing.
/// Both BM25 and embeddings benefit from the path context.
fn format_for_indexing(path: &str, content: &str) -> String {
    format!("// {path}\n{content}")
}

fn index(path: &std::path::Path) -> eyre::Result<()> {
    use indicatif::{ProgressBar, ProgressStyle};
    use rayon::prelude::*;

    let index_path = path.join(INDEX_DIR);

    // Phase 1: Discover files (parallel I/O)
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .wrap_err("invalid spinner template")?,
    );
    spinner.set_message("Discovering files...");
    spinner.enable_steady_tick(std::time::Duration::from_millis(80));

    const MAX_FILE_SIZE: u64 = 1024 * 1024; // 1MB

    // Collect paths first (fast, sequential)
    let walker = create_walker(path)?;
    let entries: Vec<_> = walker
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .collect();

    spinner.set_message(format!("Reading {} files...", entries.len()));

    // Read files in parallel
    enum FileResult {
        Ok(IndexableFile),
        SkippedLarge,
        SkippedBinary,
    }

    let results: Vec<_> = entries
        .par_iter()
        .filter_map(|entry| {
            let file_path = entry.path();

            let metadata = std::fs::metadata(file_path).ok()?;
            if metadata.len() > MAX_FILE_SIZE {
                tracing::debug!(?file_path, "skipping large file");
                return Some(FileResult::SkippedLarge);
            }

            let content_bytes = std::fs::read(file_path).ok()?;

            // Skip non-UTF8 files
            if std::str::from_utf8(&content_bytes).is_err() {
                tracing::debug!(?file_path, "skipping non-utf8 file");
                return Some(FileResult::SkippedBinary);
            }

            let relative_path = file_path
                .strip_prefix(path)
                .unwrap_or(file_path)
                .to_string_lossy()
                .to_string();

            Some(FileResult::Ok(IndexableFile {
                relative_path,
                content_bytes,
            }))
        })
        .collect();

    // Partition results
    let mut files = Vec::new();
    let mut skipped_large = 0_u64;
    let mut skipped_binary = 0_u64;

    for result in results {
        match result {
            FileResult::Ok(file) => files.push(file),
            FileResult::SkippedLarge => skipped_large += 1,
            FileResult::SkippedBinary => skipped_binary += 1,
        }
    }

    spinner.finish_with_message(format!(
        "Found {} files (skipped: {} large, {} binary)",
        files.len(),
        skipped_large,
        skipped_binary
    ));

    if files.is_empty() {
        println!("No files to index");
        return Ok(());
    }

    // Phase 2: Load encoder
    let mut encoder = load_encoder()?;

    // Phase 3: Create indices
    let bm25_path = index_path.join(BM25_DIR);
    let bm25_index = sgrep_index::Bm25Index::new_at_path(&bm25_path)
        .wrap_err_with(|| format!("failed to create BM25 index at {bm25_path:?}"))?;

    let embeddings_path = index_path.join(EMBEDDINGS_DIR);
    let mut store = sgrep_cas::EmbeddingStore::open(embeddings_path.clone())
        .wrap_err_with(|| format!("failed to create embedding store at {embeddings_path:?}"))?;

    // Phase 4: Chunk files and prepare for indexing
    let chunk_config = sgrep_chunk::ChunkConfig::default();
    let mut all_chunks: Vec<IndexableChunk> = Vec::new();
    let mut cached_count = 0_u64;

    for file in &files {
        let content_str =
            std::str::from_utf8(&file.content_bytes).expect("already verified as UTF-8");

        let chunks = sgrep_chunk::chunk_text(content_str, chunk_config);

        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let chunk_id = sgrep_chunk::chunk_id(&file.relative_path, chunk_idx);
            let formatted = format_for_indexing(&chunk_id, chunk.content);
            let hash = sgrep_cas::ContentHash::from_content(formatted.as_bytes());

            if store.has_embeddings(&hash) {
                cached_count += 1;
            }

            all_chunks.push(IndexableChunk {
                chunk_id,
                file_path: file.relative_path.clone(),
                formatted_content: formatted,
                hash,
            });
        }
    }

    // Prepare BM25 documents from chunks
    let bm25_docs: Vec<_> = all_chunks
        .iter()
        .map(|chunk| (chunk.chunk_id.clone(), chunk.formatted_content.clone()))
        .collect();

    // BM25 indexing - single batch commit (much faster than per-document)
    let bm25_spinner = ProgressBar::new_spinner();
    bm25_spinner.set_style(ProgressStyle::default_spinner().template("{spinner:.green} {msg}")?);
    bm25_spinner.set_message(format!(
        "Indexing {} chunks from {} files for BM25...",
        bm25_docs.len(),
        files.len()
    ));
    bm25_spinner.enable_steady_tick(std::time::Duration::from_millis(100));

    bm25_index
        .add_documents(bm25_docs.iter().map(|(p, c)| (p.as_str(), c.as_str())))
        .wrap_err("failed to add documents to BM25 index")?;

    bm25_spinner.finish_with_message(format!(
        "BM25 index: {} chunks from {} files",
        all_chunks.len(),
        files.len()
    ));

    // Phase 5: Batched GPU embedding
    let mut embedded_count = 0_u64;
    let mut encode_errors = 0_u64;
    let start_time = std::time::Instant::now();

    // Filter to only chunks that need embedding
    let chunks_to_embed: Vec<_> = all_chunks
        .iter()
        .filter(|chunk| !store.has_embeddings(&chunk.hash))
        .collect();

    if !chunks_to_embed.is_empty() {
        // Dynamic batching: pack chunks by estimated token count
        const MAX_BATCH_FILES: usize = 64;
        const MAX_BATCH_TOKENS: usize = 8192;
        const CHARS_PER_TOKEN: usize = 4; // rough estimate

        let embed_progress = ProgressBar::new(chunks_to_embed.len() as u64);
        embed_progress.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}) Embedding: {msg}")
                .wrap_err("invalid progress bar template")?
                .progress_chars("█▓░"),
        );

        let mut batch_start = 0;
        while batch_start < chunks_to_embed.len() {
            // Build batch dynamically based on estimated token count
            let mut batch_end = batch_start;
            let mut batch_tokens = 0;

            while batch_end < chunks_to_embed.len() {
                let chunk = chunks_to_embed[batch_end];
                let est_tokens = chunk.formatted_content.len() / CHARS_PER_TOKEN;

                // Stop if adding this chunk would exceed limits (unless batch is empty)
                if batch_end > batch_start
                    && (batch_end - batch_start >= MAX_BATCH_FILES
                        || batch_tokens + est_tokens > MAX_BATCH_TOKENS)
                {
                    break;
                }

                batch_tokens += est_tokens;
                batch_end += 1;
            }

            let batch = &chunks_to_embed[batch_start..batch_end];
            let texts: Vec<&str> = batch.iter().map(|c| c.formatted_content.as_str()).collect();

            // Show batch info
            if let Some(first) = batch.first() {
                embed_progress
                    .set_message(format!("{} (+{}, ~{}tok)", first.chunk_id, batch.len().saturating_sub(1), batch_tokens));
            }

            match encoder.encode_documents_batch(&texts) {
                Ok(embeddings) => {
                    for (chunk, embedding) in batch.iter().zip(embeddings.into_iter()) {
                        store
                            .store_embeddings(&chunk.hash, embedding.view())
                            .wrap_err_with(|| format!("failed to store embeddings for {}", chunk.chunk_id))?;
                        embedded_count += 1;
                    }
                }
                Err(e) => {
                    // Batch failed, try individually to isolate bad chunks
                    tracing::warn!(?e, "batch encoding failed, falling back to individual encoding");
                    for chunk in batch {
                        match encoder.encode_document(&chunk.formatted_content) {
                            Ok(embedding) => {
                                store.store_embeddings(&chunk.hash, embedding.view())?;
                                embedded_count += 1;
                            }
                            Err(e) => {
                                encode_errors += 1;
                                tracing::warn!(path = %chunk.chunk_id, ?e, "failed to encode chunk");
                            }
                        }
                    }
                }
            }

            embed_progress.inc(batch.len() as u64);
            batch_start = batch_end;
        }
        embed_progress.finish_and_clear();
    }

    let elapsed = start_time.elapsed();

    // Final summary
    let total_files = files.len();
    let total_chunks = all_chunks.len();
    println!("Indexed {total_files} files ({total_chunks} chunks)");

    if embedded_count > 0 {
        let embed_per_sec = embedded_count as f64 / elapsed.as_secs_f64();
        println!(
            "  Embeddings: {} new in {:.1}s ({:.1}/sec), {} cached",
            embedded_count,
            elapsed.as_secs_f64(),
            embed_per_sec,
            cached_count
        );
    } else if cached_count > 0 {
        println!("  Embeddings: {} cached (all up to date)", cached_count);
    }

    if encode_errors > 0 {
        println!("  Encode errors: {encode_errors}");
    }

    Ok(())
}
