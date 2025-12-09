//! Benchmarks for MaxSim computation.

use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn random_embedding(num_tokens: usize, dim: usize) -> sgrep_core::DocumentEmbedding {
    let embeddings: Vec<Vec<f32>> = (0..num_tokens)
        .map(|i| (0..dim).map(|j| ((i * dim + j) as f32).sin()).collect())
        .collect();
    sgrep_core::DocumentEmbedding::new(embeddings, dim)
}

fn bench_maxsim(c: &mut Criterion) {
    let query = random_embedding(32, 768); // Typical query length
    let doc = random_embedding(512, 768); // Typical document length

    c.bench_function("maxsim_32x512_dim768", |b| {
        b.iter(|| sgrep_embed::maxsim(black_box(&query), black_box(&doc)).unwrap())
    });

    let short_query = random_embedding(8, 768);
    let short_doc = random_embedding(128, 768);

    c.bench_function("maxsim_8x128_dim768", |b| {
        b.iter(|| sgrep_embed::maxsim(black_box(&short_query), black_box(&short_doc)).unwrap())
    });
}

criterion_group!(benches, bench_maxsim);
criterion_main!(benches);
