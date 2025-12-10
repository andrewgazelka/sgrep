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

/// Create a file walker that respects gitignore and always ignores .sgrep.
fn create_walker(path: &std::path::Path) -> eyre::Result<ignore::Walk> {
    use eyre::WrapErr as _;

    let mut builder = ignore::WalkBuilder::new(path);
    builder
        .hidden(false)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true);

    // Always ignore .sgrep directory
    let mut overrides = ignore::overrides::OverrideBuilder::new(path);
    overrides
        .add("!.sgrep/")
        .wrap_err("failed to add .sgrep override")?;
    builder.overrides(overrides.build().wrap_err("failed to build overrides")?);

    Ok(builder.build())
}

/// Build a mapping from doc_id to content hash for semantic search.
fn build_doc_hash_map(
    store: &sgrep_cas::EmbeddingStore,
    path: &std::path::Path,
) -> eyre::Result<std::collections::HashMap<String, sgrep_cas::ContentHash>> {
    let mut map = std::collections::HashMap::new();

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

        let hash = sgrep_cas::ContentHash::from_content(&content);
        if store.has_embeddings(&hash) {
            let relative_path = file_path
                .strip_prefix(path)
                .unwrap_or(file_path)
                .to_string_lossy()
                .to_string();
            map.insert(relative_path, hash);
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

    // BM25 search
    let bm25_path = index_path.join(BM25_DIR);
    let bm25_index =
        sgrep_index::Bm25Index::new_at_path(&bm25_path).wrap_err("failed to open BM25 index")?;

    // Get more results for fusion (we'll re-rank)
    let fetch_limit = limit * 3;
    let bm25_results = bm25_index
        .search(query, fetch_limit)
        .wrap_err("BM25 search failed")?;

    // Build BM25 score map
    let bm25_scores: std::collections::HashMap<String, f32> = bm25_results
        .iter()
        .map(|r| (r.doc_id.clone(), r.score))
        .collect();

    if bm25_only || bm25_results.is_empty() {
        if json {
            let results: Vec<JsonResult> = bm25_results
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
            for result in bm25_results.iter().take(limit) {
                println!("{}: {:.4}", result.doc_id, result.score);
            }
        }
        return Ok(());
    }

    // Semantic search with ColBERT
    let embeddings_path = index_path.join(EMBEDDINGS_DIR);
    let mut store = sgrep_cas::EmbeddingStore::open(embeddings_path)
        .wrap_err("failed to open embedding store")?;

    // Build doc_id -> hash mapping
    let doc_hash_map = build_doc_hash_map(&store, path).wrap_err("failed to build doc hash map")?;

    if doc_hash_map.is_empty() {
        // No embeddings, fall back to BM25 only
        eprintln!("Warning: no embeddings found, using BM25 only");
        if json {
            let results: Vec<JsonResult> = bm25_results
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
            for result in bm25_results.iter().take(limit) {
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

    // Collect document embeddings for batch processing
    let mut doc_ids = Vec::new();
    let mut doc_hashes = Vec::new();

    for bm25_result in &bm25_results {
        if let Some(hash) = doc_hash_map.get(&bm25_result.doc_id) {
            doc_ids.push(bm25_result.doc_id.clone());
            doc_hashes.push(hash);
        }
    }

    // Get batch views and compute MaxSim scores on GPU
    let gpu = sgrep_embed::GpuDevice::new().wrap_err("failed to create GPU device")?;

    let batch_views = store
        .get_batch_views(&doc_hashes)
        .wrap_err("failed to get batch views")?;

    // Extract just the views for batch processing
    let doc_views: Vec<_> = batch_views.iter().map(|(_, v)| *v).collect();

    let colbert_scores_vec = gpu
        .batch_maxsim(query_embedding.view(), &doc_views)
        .wrap_err("failed to compute batch MaxSim")?;

    // Build score maps
    let mut colbert_scores: std::collections::HashMap<String, f32> =
        std::collections::HashMap::new();
    let mut semantic_results = Vec::new();

    for (i, (hash, _)) in batch_views.iter().enumerate() {
        // Find doc_id for this hash
        let doc_id = doc_hashes
            .iter()
            .zip(doc_ids.iter())
            .find(|(h, _)| **h == hash)
            .map(|(_, id)| id.clone());

        if let Some(doc_id) = doc_id {
            let score = colbert_scores_vec[i];
            colbert_scores.insert(doc_id.clone(), score);
            semantic_results.push(sgrep_core::SearchResult { doc_id, score });
        }
    }

    // Fuse results using RRF
    let fused = sgrep_fusion::reciprocal_rank_fusion(&[bm25_results, semantic_results]);

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

fn index(path: &std::path::Path) -> eyre::Result<()> {
    use indicatif::{ProgressBar, ProgressStyle};

    let index_path = path.join(INDEX_DIR);

    // Phase 1: Discover files
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .wrap_err("invalid spinner template")?,
    );
    spinner.set_message("Discovering files...");
    spinner.enable_steady_tick(std::time::Duration::from_millis(80));

    let walker = create_walker(path)?;

    const MAX_FILE_SIZE: u64 = 1024 * 1024; // 1MB
    let mut files: Vec<IndexableFile> = Vec::new();
    let mut skipped_large = 0_u64;
    let mut skipped_binary = 0_u64;

    for entry in walker {
        let entry = entry.wrap_err("failed to read directory entry")?;
        let file_path = entry.path();

        if !file_path.is_file() {
            continue;
        }

        let metadata = std::fs::metadata(file_path)
            .wrap_err_with(|| format!("failed to read metadata for {file_path:?}"))?;

        if metadata.len() > MAX_FILE_SIZE {
            skipped_large += 1;
            tracing::debug!(?file_path, "skipping large file");
            continue;
        }

        let content_bytes = match std::fs::read(file_path) {
            Ok(c) => c,
            Err(_) => {
                tracing::debug!(?file_path, "skipping unreadable file");
                continue;
            }
        };

        // Skip non-UTF8 files
        if std::str::from_utf8(&content_bytes).is_err() {
            skipped_binary += 1;
            tracing::debug!(?file_path, "skipping non-utf8 file");
            continue;
        }

        let relative_path = file_path
            .strip_prefix(path)
            .unwrap_or(file_path)
            .to_string_lossy()
            .to_string();

        spinner.set_message(format!("Discovering files... {}", files.len()));
        files.push(IndexableFile {
            relative_path,
            content_bytes,
        });
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

    // Phase 4: Index with progress bar
    let progress = ProgressBar::new(files.len() as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}) {msg}")
            .wrap_err("invalid progress bar template")?
            .progress_chars("█▓░"),
    );

    let mut embedded_count = 0_u64;
    let mut cached_count = 0_u64;
    let mut encode_errors = 0_u64;
    let start_time = std::time::Instant::now();

    for file in &files {
        // Safe: we verified UTF-8 above
        let content_str = std::str::from_utf8(&file.content_bytes)
            .expect("already verified as UTF-8");

        // Index in BM25
        bm25_index
            .add_document(&file.relative_path, content_str)
            .wrap_err_with(|| format!("failed to add {} to BM25 index", file.relative_path))?;

        // Compute and store embeddings (skip if already cached)
        let hash = sgrep_cas::ContentHash::from_content(&file.content_bytes);
        if !store.has_embeddings(&hash) {
            match encoder.encode_document(content_str) {
                Ok(embedding) => {
                    store
                        .store_embeddings(&hash, embedding.view())
                        .wrap_err_with(|| {
                            format!("failed to store embeddings for {}", file.relative_path)
                        })?;
                    embedded_count += 1;
                }
                Err(e) => {
                    encode_errors += 1;
                    tracing::warn!(path = %file.relative_path, ?e, "failed to encode document");
                }
            }
        } else {
            cached_count += 1;
        }

        progress.set_message(file.relative_path.clone());
        progress.inc(1);
    }

    let elapsed = start_time.elapsed();
    progress.finish_and_clear();

    // Final summary
    let total = files.len() as u64;
    let files_per_sec = if elapsed.as_secs_f64() > 0.0 {
        total as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    println!(
        "Indexed {} files in {:.1}s ({:.1} files/sec)",
        total,
        elapsed.as_secs_f64(),
        files_per_sec
    );
    println!(
        "  Embeddings: {} new, {} cached",
        embedded_count, cached_count
    );
    if encode_errors > 0 {
        println!("  Encode errors: {encode_errors}");
    }

    Ok(())
}
