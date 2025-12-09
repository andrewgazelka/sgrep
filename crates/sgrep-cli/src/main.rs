//! CLI for semantic grep.

// CLI binaries need to print user-facing output
#![allow(
    clippy::print_stdout,
    reason = "CLI binary needs stdout for user output"
)]

use eyre::WrapErr as _;

const MODEL_REPO: &str = "jinaai/jina-colbert-v2";
const BM25_DIR: &str = "bm25";
const CAS_DIR: &str = "cas";
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
        Command::Index { path, verbose } => {
            index(&path, verbose)?;
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

        /// Show verbose output (files being indexed)
        #[arg(short, long)]
        verbose: bool,
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

/// Build a mapping from doc_id to content hash for semantic search.
fn build_doc_hash_map(
    cas: &sgrep_cas::CasStore,
    path: &std::path::Path,
) -> eyre::Result<std::collections::HashMap<String, sgrep_cas::ContentHash>> {
    let mut map = std::collections::HashMap::new();

    let walker = ignore::WalkBuilder::new(path)
        .hidden(false)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .build();

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
        if cas.has_embeddings(&hash) {
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
    let cas_path = index_path.join(CAS_DIR);
    let cas = sgrep_cas::CasStore::new(cas_path).wrap_err("failed to open CAS store")?;

    // Build doc_id -> hash mapping
    let doc_hash_map = build_doc_hash_map(&cas, path).wrap_err("failed to build doc hash map")?;

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

    // Compute MaxSim scores for BM25 candidates
    let mut semantic_results = Vec::new();
    let mut colbert_scores: std::collections::HashMap<String, f32> =
        std::collections::HashMap::new();

    for bm25_result in &bm25_results {
        let Some(hash) = doc_hash_map.get(&bm25_result.doc_id) else {
            continue;
        };

        let Some(doc_embedding) = cas
            .load_embeddings(hash)
            .wrap_err("failed to load embeddings")?
        else {
            continue;
        };

        let score = sgrep_embed::maxsim(&query_embedding, &doc_embedding)
            .wrap_err("failed to compute MaxSim")?;

        colbert_scores.insert(bm25_result.doc_id.clone(), score);
        semantic_results.push(sgrep_core::SearchResult {
            doc_id: bm25_result.doc_id.clone(),
            score,
        });
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

fn index(path: &std::path::Path, verbose: bool) -> eyre::Result<()> {
    let index_path = path.join(INDEX_DIR);
    eprintln!("Indexing {} -> {}", path.display(), index_path.display());

    // Create BM25 index
    let bm25_path = index_path.join(BM25_DIR);
    let bm25_index = sgrep_index::Bm25Index::new_at_path(&bm25_path)
        .wrap_err_with(|| format!("failed to create BM25 index at {bm25_path:?}"))?;

    // Create CAS store for embeddings
    let cas_path = index_path.join(CAS_DIR);
    let cas = sgrep_cas::CasStore::new(cas_path.clone())
        .wrap_err_with(|| format!("failed to create CAS store at {cas_path:?}"))?;

    // Load ColBERT encoder
    let mut encoder = load_encoder()?;

    // Collect files to index
    let walker = ignore::WalkBuilder::new(path)
        .hidden(false)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .build();

    let mut count = 0;
    let mut embedded_count = 0;

    for entry in walker {
        let entry = entry.wrap_err("failed to read directory entry")?;
        let file_path = entry.path();

        if !file_path.is_file() {
            continue;
        }

        // Skip binary files and large files
        let metadata = std::fs::metadata(file_path)
            .wrap_err_with(|| format!("failed to read metadata for {file_path:?}"))?;

        const MAX_FILE_SIZE: u64 = 1024 * 1024; // 1MB
        if metadata.len() > MAX_FILE_SIZE {
            tracing::debug!(?file_path, "skipping large file");
            continue;
        }

        // Read file content
        let content_bytes = match std::fs::read(file_path) {
            Ok(c) => c,
            Err(_) => {
                tracing::debug!(?file_path, "skipping unreadable file");
                continue;
            }
        };

        // Try to read as UTF-8 for BM25
        let content_str = match std::str::from_utf8(&content_bytes) {
            Ok(s) => s,
            Err(_) => {
                tracing::debug!(?file_path, "skipping non-utf8 file");
                continue;
            }
        };

        let relative_path = file_path
            .strip_prefix(path)
            .unwrap_or(file_path)
            .to_string_lossy();

        // Index in BM25
        bm25_index
            .add_document(&relative_path, content_str)
            .wrap_err_with(|| format!("failed to add {relative_path} to BM25 index"))?;

        // Compute and store embeddings (skip if already cached)
        let hash = sgrep_cas::ContentHash::from_content(&content_bytes);
        if !cas.has_embeddings(&hash) {
            match encoder.encode_document(content_str) {
                Ok(embedding) => {
                    cas.store_embeddings(&hash, &embedding).wrap_err_with(|| {
                        format!("failed to store embeddings for {relative_path}")
                    })?;
                    embedded_count += 1;
                }
                Err(e) => {
                    tracing::warn!(?file_path, ?e, "failed to encode document");
                }
            }
        } else {
            embedded_count += 1;
        }

        count += 1;
        if verbose {
            eprintln!("  {relative_path}");
        } else if count % 100 == 0 {
            eprintln!("Indexed {count} files...");
        }
    }

    println!("Indexed {count} files ({embedded_count} with embeddings)");

    Ok(())
}
