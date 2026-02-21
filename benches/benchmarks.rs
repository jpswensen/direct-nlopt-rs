//! Benchmarks for direct-nlopt-rs
//! These will be populated as the implementation progresses.

use criterion::{criterion_group, criterion_main, Criterion};

fn placeholder_benchmark(c: &mut Criterion) {
    c.bench_function("placeholder", |b| {
        b.iter(|| {
            // Will be replaced with real benchmarks
            std::hint::black_box(42)
        })
    });
}

criterion_group!(benches, placeholder_benchmark);
criterion_main!(benches);
