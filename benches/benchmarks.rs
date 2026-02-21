//! Benchmarks for direct-nlopt-rs
//!
//! Includes parallel evaluation threshold benchmarks to determine the optimal
//! `min_parallel_evals` setting.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use direct_nlopt::{DirectBuilder, types::DirectAlgorithm};

/// Sphere function: f(x) = sum(x_i^2)
fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

/// Expensive sphere: adds artificial computation per evaluation to simulate
/// a costly objective function (~10µs per eval).
fn expensive_sphere(x: &[f64]) -> f64 {
    let base: f64 = x.iter().map(|xi| xi * xi).sum();
    let mut acc = base;
    for i in 0..1000 {
        acc += (acc + i as f64).sin() * 0.0001;
    }
    acc - (acc - base) // cancel out the noise, keep the cost
}

fn bench_parallel_threshold_cheap(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_threshold_cheap_5d");
    group.sample_size(10);

    let bounds = vec![(-5.0, 5.0); 5];

    // Serial baseline
    group.bench_function("serial", |b| {
        b.iter(|| {
            DirectBuilder::new(sphere, bounds.clone())
                .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
                .max_feval(2000)
                .parallel(false)
                .minimize()
                .unwrap()
        })
    });

    // Test different thresholds
    for threshold in [1, 2, 4, 8, 16, 32] {
        group.bench_with_input(
            BenchmarkId::new("threshold", threshold),
            &threshold,
            |b, &t| {
                b.iter(|| {
                    DirectBuilder::new(sphere, bounds.clone())
                        .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
                        .max_feval(2000)
                        .parallel(true)
                        .min_parallel_evals(t)
                        .minimize()
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_parallel_threshold_expensive(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_threshold_expensive_5d");
    group.sample_size(10);

    let bounds = vec![(-5.0, 5.0); 5];

    // Serial baseline
    group.bench_function("serial", |b| {
        b.iter(|| {
            DirectBuilder::new(expensive_sphere, bounds.clone())
                .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
                .max_feval(2000)
                .parallel(false)
                .minimize()
                .unwrap()
        })
    });

    // Test different thresholds
    for threshold in [1, 2, 4, 8, 16, 32] {
        group.bench_with_input(
            BenchmarkId::new("threshold", threshold),
            &threshold,
            |b, &t| {
                b.iter(|| {
                    DirectBuilder::new(expensive_sphere, bounds.clone())
                        .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
                        .max_feval(2000)
                        .parallel(true)
                        .min_parallel_evals(t)
                        .minimize()
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_parallel_threshold_cheap,
    bench_parallel_threshold_expensive
);
criterion_main!(benches);
