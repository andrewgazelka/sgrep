//! Benchmarks for MaxSim computation.

use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn random_embedding(num_tokens: usize, dim: usize) -> sgrep_core::Embedding {
    ndarray::Array2::from_shape_fn((num_tokens, dim), |(i, j)| ((i * dim + j) as f32).sin())
}

fn bench_maxsim(c: &mut Criterion) {
    let query = random_embedding(32, sgrep_core::EMBEDDING_DIM);
    let doc = random_embedding(512, sgrep_core::EMBEDDING_DIM);

    c.bench_function("maxsim_32x512_dim128", |b| {
        b.iter(|| sgrep_embed::maxsim(black_box(query.view()), black_box(doc.view())).unwrap())
    });

    let short_query = random_embedding(8, sgrep_core::EMBEDDING_DIM);
    let short_doc = random_embedding(128, sgrep_core::EMBEDDING_DIM);

    c.bench_function("maxsim_8x128_dim128", |b| {
        b.iter(|| {
            sgrep_embed::maxsim(black_box(short_query.view()), black_box(short_doc.view())).unwrap()
        })
    });
}

criterion_group!(benches, bench_maxsim);
criterion_main!(benches);
