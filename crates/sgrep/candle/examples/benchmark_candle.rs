//! Benchmark comparing Candle Metal vs ONNX Runtime CPU.
//!
//! Run with: cargo run --release --example benchmark_candle --features metal

use std::io::Write as _;

use eyre::WrapErr as _;

const MODEL_DIR: &str = "scripts/convert/models";
const WARMUP_ITERATIONS: usize = 3;
const BENCHMARK_ITERATIONS: usize = 20;

const TEST_TEXTS: &[&str] = &[
    "Hello, world!",
    "fn main() { println!(\"Hello\"); }",
    "This is a test of the semantic grep system for searching code.",
    "pub struct ColBertEncoder { session: Session, tokenizer: Tokenizer }",
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
];

fn benchmark_device(
    device_name: &str,
    device: candle_core::Device,
    results: &mut Vec<(String, f64, f64)>,
) -> eyre::Result<()> {
    tracing::info!(%device_name, "loading encoder");

    let model_path = format!("{MODEL_DIR}/model.safetensors");
    let tokenizer_path = format!("{MODEL_DIR}/tokenizer.json");
    let config_path = format!("{MODEL_DIR}/config.json");

    let mut encoder = sgrep_candle::ColBertEncoder::load_with_device(
        &model_path,
        &tokenizer_path,
        &config_path,
        device,
    )
    .wrap_err_with(|| format!("failed to load encoder with {device_name}"))?;

    // Warmup
    tracing::info!(%device_name, "warming up ({WARMUP_ITERATIONS} iterations)");
    for _ in 0..WARMUP_ITERATIONS {
        for text in TEST_TEXTS {
            let _ = encoder.encode_document(text)?;
        }
    }

    // Benchmark
    tracing::info!(%device_name, "benchmarking ({BENCHMARK_ITERATIONS} iterations)");
    let mut latencies = Vec::with_capacity(BENCHMARK_ITERATIONS * TEST_TEXTS.len());

    for _ in 0..BENCHMARK_ITERATIONS {
        for text in TEST_TEXTS {
            let start = std::time::Instant::now();
            let _ = encoder.encode_document(text)?;
            latencies.push(start.elapsed().as_secs_f64() * 1000.0); // ms
        }
    }

    // Calculate stats
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let p50 = latencies[latencies.len() / 2];
    let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];
    let min = latencies.first().copied().unwrap_or(0.0);
    let max = latencies.last().copied().unwrap_or(0.0);

    tracing::info!(
        %device_name,
        mean_ms = format!("{mean:.2}"),
        p50_ms = format!("{p50:.2}"),
        p95_ms = format!("{p95:.2}"),
        p99_ms = format!("{p99:.2}"),
        min_ms = format!("{min:.2}"),
        max_ms = format!("{max:.2}"),
        "benchmark complete"
    );

    results.push((device_name.to_string(), mean, p50));

    Ok(())
}

fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    tracing::info!("Starting Candle benchmark");
    tracing::info!(
        warmup = WARMUP_ITERATIONS,
        iterations = BENCHMARK_ITERATIONS,
        texts = TEST_TEXTS.len(),
        "configuration"
    );

    let mut results = Vec::new();

    // Benchmark CPU
    tracing::info!("benchmarking CPU...");
    if let Err(e) = benchmark_device("Candle-CPU", sgrep_candle::cpu_device(), &mut results) {
        tracing::error!(error = ?e, "CPU benchmark failed");
    }

    // Benchmark Metal (if available)
    #[cfg(feature = "metal")]
    {
        tracing::info!("benchmarking Metal...");
        match sgrep_candle::default_device() {
            Ok(device) => {
                if let Err(e) = benchmark_device("Candle-Metal", device, &mut results) {
                    tracing::error!(error = ?e, "Metal benchmark failed");
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "failed to create Metal device");
            }
        }
    }

    // Write results to file
    let output_path = "candle_benchmark_results.txt";
    let mut file = std::fs::File::create(output_path).wrap_err("failed to create results file")?;

    writeln!(file, "Candle Benchmark Results")?;
    writeln!(file, "========================")?;
    writeln!(file)?;
    writeln!(
        file,
        "Configuration: {WARMUP_ITERATIONS} warmup, {BENCHMARK_ITERATIONS} iterations, {} texts",
        TEST_TEXTS.len()
    )?;
    writeln!(file)?;
    writeln!(
        file,
        "{:<20} {:>12} {:>12}",
        "Provider", "Mean (ms)", "P50 (ms)"
    )?;
    writeln!(file, "{:-<20} {:->12} {:->12}", "", "", "")?;

    for (name, mean, p50) in &results {
        writeln!(file, "{name:<20} {mean:>12.2} {p50:>12.2}")?;
    }

    writeln!(file)?;

    // Calculate speedups relative to CPU
    if let Some((_, cpu_mean, _)) = results.iter().find(|(n, _, _)| n == "Candle-CPU") {
        writeln!(file, "Speedup vs Candle-CPU:")?;
        for (name, mean, _) in &results {
            if name != "Candle-CPU" {
                let speedup = cpu_mean / mean;
                writeln!(file, "  {name}: {speedup:.2}x")?;
            }
        }
    }

    tracing::info!(%output_path, "results written to file");

    // Print summary
    eprintln!("\n=== CANDLE BENCHMARK SUMMARY ===\n");
    eprintln!("{:<20} {:>12} {:>12}", "Provider", "Mean (ms)", "P50 (ms)");
    eprintln!("{:-<20} {:->12} {:->12}", "", "", "");
    for (name, mean, p50) in &results {
        eprintln!("{name:<20} {mean:>12.2} {p50:>12.2}");
    }

    Ok(())
}
