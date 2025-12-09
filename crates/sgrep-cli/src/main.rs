//! CLI for semantic grep.

fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    match args.command {
        Command::Search { query, path, limit } => {
            search(&query, &path, limit)?;
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
    },

    /// Index a directory for searching
    Index {
        /// Path to index
        path: std::path::PathBuf,
    },
}

fn search(query: &str, path: &std::path::Path, limit: usize) -> eyre::Result<()> {
    use eyre::WrapErr as _;

    tracing::info!(?query, ?path, limit, "searching");

    // For now, just use BM25
    let index_path = path.join(".sgrep");
    if !index_path.exists() {
        tracing::warn!("no index found, run `sgrep index` first");
        eyre::bail!("no index found at {index_path:?}, run `sgrep index {path:?}` first");
    }

    let index =
        sgrep_index::Bm25Index::new_at_path(&index_path).wrap_err("failed to open index")?;

    let results = index.search(query, limit).wrap_err("search failed")?;

    for result in results {
        // Using tracing for output since println is disallowed
        tracing::info!(path = %result.doc_id, score = result.score);
    }

    Ok(())
}

fn index(path: &std::path::Path) -> eyre::Result<()> {
    use eyre::WrapErr as _;

    tracing::info!(?path, "indexing");

    let index_path = path.join(".sgrep");
    let index = sgrep_index::Bm25Index::new_at_path(&index_path)
        .wrap_err_with(|| format!("failed to create index at {index_path:?}"))?;

    // Use ignore crate to respect .gitignore
    let walker = ignore::WalkBuilder::new(path)
        .hidden(false)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .build();

    let mut count = 0;
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

        // Try to read as UTF-8
        let content = match std::fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(_) => {
                tracing::debug!(?file_path, "skipping non-utf8 file");
                continue;
            }
        };

        let relative_path = file_path
            .strip_prefix(path)
            .unwrap_or(file_path)
            .to_string_lossy();

        index
            .add_document(&relative_path, &content)
            .wrap_err_with(|| format!("failed to index {file_path:?}"))?;

        count += 1;
        if count % 100 == 0 {
            tracing::info!(count, "indexed files");
        }
    }

    tracing::info!(count, "indexing complete");

    Ok(())
}
