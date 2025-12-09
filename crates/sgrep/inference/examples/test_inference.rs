//! Test inference with the exported ONNX model.

fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let model_path = "scripts/convert/models/jina-colbert-v2.onnx";
    let tokenizer_path = "scripts/convert/models/tokenizer.json";

    tracing::info!("Loading encoder...");
    let mut encoder = sgrep_inference::ColBertEncoder::load(model_path, tokenizer_path)?;

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
            let preview: Vec<_> = first_emb.iter().take(5).collect();
            tracing::info!(?preview, "First token embedding (first 5 values)");
        }
    }

    // Test MaxSim scoring between two embeddings
    tracing::info!("Testing MaxSim scoring...");
    let query_emb = encoder.encode("function main")?;
    let doc_emb = encoder.encode("fn main() { println!(\"Hello\"); }")?;

    let score = sgrep_embed::maxsim(&query_emb, &doc_emb)?;
    tracing::info!(?score, "MaxSim score between query and document");

    tracing::info!("Test completed successfully!");
    Ok(())
}
