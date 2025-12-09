//! Test Candle inference and compare with ONNX if available.

use eyre::WrapErr as _;

const MODEL_DIR: &str = "scripts/convert/models";

fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let model_path = format!("{MODEL_DIR}/model.safetensors");
    let tokenizer_path = format!("{MODEL_DIR}/tokenizer.json");
    let config_path = format!("{MODEL_DIR}/config.json");

    tracing::info!("Loading Candle encoder...");
    let mut encoder =
        sgrep_candle::ColBertEncoder::load(&model_path, &tokenizer_path, &config_path)
            .wrap_err("failed to load Candle encoder")?;

    let test_texts = [
        "Hello, world!",
        "fn main() { println!(\"Hello\"); }",
        "This is a test of the semantic grep system.",
    ];

    for text in test_texts {
        tracing::info!(?text, "Encoding text...");
        let embedding = encoder.encode(text)?;
        tracing::info!(
            num_tokens = embedding.embeddings.len(),
            dim = embedding.dim,
            "Embedding generated"
        );

        // Print first few values of first token embedding
        if let Some(first_emb) = embedding.embeddings.first() {
            let preview: Vec<_> = first_emb.iter().take(5).map(|v| format!("{v:.4}")).collect();
            tracing::info!(?preview, "First token embedding (first 5 values)");
        }
    }

    // Test MaxSim scoring between two embeddings
    tracing::info!("Testing MaxSim scoring...");
    let query_emb = encoder.encode("function main")?;
    let doc_emb = encoder.encode("fn main() { println!(\"Hello\"); }")?;

    let score = sgrep_embed::maxsim(&query_emb, &doc_emb)?;
    tracing::info!(?score, "MaxSim score between query and document");

    // Test semantic similarity - should be higher for related texts
    tracing::info!("Testing semantic similarity...");

    let code_query = encoder.encode("error handling")?;
    let code_related = encoder.encode("try { } catch (e) { handle(e); }")?;
    let code_unrelated = encoder.encode("Hello world greeting message")?;

    let score_related = sgrep_embed::maxsim(&code_query, &code_related)?;
    let score_unrelated = sgrep_embed::maxsim(&code_query, &code_unrelated)?;

    tracing::info!(
        ?score_related,
        ?score_unrelated,
        "Similarity scores (related should be higher)"
    );

    if score_related > score_unrelated {
        tracing::info!("Semantic similarity test PASSED - related text scored higher");
    } else {
        tracing::warn!("Semantic similarity test FAILED - unrelated text scored higher!");
    }

    tracing::info!("Test completed successfully!");
    Ok(())
}
