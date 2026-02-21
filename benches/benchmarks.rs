//! Benchmarks for direct-nlopt-rs
//!
//! Comprehensive benchmark suite covering:
//! - Standard test functions: sphere, rosenbrock, rastrigin
//! - Dimensions: 2D, 5D, 10D
//! - Algorithm variants: GablonskyOriginal, GablonskyLocallyBiased
//! - Parallel modes: serial (apples-to-apples vs C), parallel
//! - Parallel threshold tuning

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use direct_nlopt::{DirectBuilder, types::DirectAlgorithm};

/// Sphere function: f(x) = sum(x_i^2)
fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

/// Rosenbrock function: f(x) = sum(100*(x_{i+1}-x_i^2)^2 + (1-x_i)^2)
fn rosenbrock(x: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..x.len() - 1 {
        let t1 = x[i + 1] - x[i] * x[i];
        let t2 = 1.0 - x[i];
        sum += 100.0 * t1 * t1 + t2 * t2;
    }
    sum
}

/// Rastrigin function: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mut sum = 10.0 * n;
    for xi in x {
        sum += xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos();
    }
    sum
}

/// Expensive sphere: adds artificial computation per evaluation to simulate
/// a costly objective function.
fn expensive_sphere(x: &[f64]) -> f64 {
    let base: f64 = x.iter().map(|xi| xi * xi).sum();
    let mut acc = base;
    for i in 0..1000 {
        acc += (acc + i as f64).sin() * 0.0001;
    }
    acc - (acc - base)
}

/// Expensive Rosenbrock with 10,000 sin/cos iterations per evaluation.
fn expensive_rosenbrock(x: &[f64]) -> f64 {
    let mut extra_work = 0.0;
    for _ in 0..10_000 {
        for xi in x.iter() {
            extra_work += (xi.sin() * xi.cos()).abs();
        }
    }
    let mut result = 0.0;
    for i in 0..x.len() - 1 {
        let t1 = x[i + 1] - x[i] * x[i];
        let t2 = 1.0 - x[i];
        result += 100.0 * t1 * t1 + t2 * t2;
    }
    result + extra_work * 1e-20
}

/// Expensive Rastrigin with 10,000 sin/cos iterations per evaluation.
fn expensive_rastrigin(x: &[f64]) -> f64 {
    let mut extra_work = 0.0;
    for _ in 0..10_000 {
        for xi in x.iter() {
            extra_work += (xi.sin() * xi.cos()).abs();
        }
    }
    let a = 10.0;
    let n = x.len() as f64;
    let result = a * n
        + x.iter()
            .map(|&xi| xi * xi - a * (2.0 * std::f64::consts::PI * xi).cos())
            .sum::<f64>();
    result + extra_work * 1e-20
}

// ── Standard test function benchmarks (matching nlopt_bench.c) ──

fn bench_sphere(c: &mut Criterion) {
    let mut group = c.benchmark_group("sphere");
    group.sample_size(10);

    for &dim in &[2, 5, 10] {
        let bounds = vec![(-5.0, 5.0); dim];

        // Gablonsky (DIRECT-L) — serial
        group.bench_with_input(
            BenchmarkId::new("gablonsky_serial", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    DirectBuilder::new(sphere, bounds.clone())
                        .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
                        .max_feval(5000)
                        .parallel(false)
                        .minimize()
                        .unwrap()
                })
            },
        );

        // Gablonsky (DIRECT-L) — parallel
        group.bench_with_input(
            BenchmarkId::new("gablonsky_parallel", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    DirectBuilder::new(sphere, bounds.clone())
                        .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
                        .max_feval(5000)
                        .parallel(true)
                        .minimize()
                        .unwrap()
                })
            },
        );

        // Original (Jones) — serial
        group.bench_with_input(
            BenchmarkId::new("original_serial", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    DirectBuilder::new(sphere, bounds.clone())
                        .algorithm(DirectAlgorithm::GablonskyOriginal)
                        .max_feval(5000)
                        .parallel(false)
                        .minimize()
                        .unwrap()
                })
            },
        );

        // Original (Jones) — parallel
        group.bench_with_input(
            BenchmarkId::new("original_parallel", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    DirectBuilder::new(sphere, bounds.clone())
                        .algorithm(DirectAlgorithm::GablonskyOriginal)
                        .max_feval(5000)
                        .parallel(true)
                        .minimize()
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_rosenbrock(c: &mut Criterion) {
    let mut group = c.benchmark_group("rosenbrock");
    group.sample_size(10);

    for &dim in &[2, 5] {
        let bounds = vec![(-5.0, 5.0); dim];

        // Gablonsky — serial
        group.bench_with_input(
            BenchmarkId::new("gablonsky_serial", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    DirectBuilder::new(rosenbrock, bounds.clone())
                        .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
                        .max_feval(5000)
                        .parallel(false)
                        .minimize()
                        .unwrap()
                })
            },
        );

        // Gablonsky — parallel
        group.bench_with_input(
            BenchmarkId::new("gablonsky_parallel", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    DirectBuilder::new(rosenbrock, bounds.clone())
                        .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
                        .max_feval(5000)
                        .parallel(true)
                        .minimize()
                        .unwrap()
                })
            },
        );

        // Original — serial
        group.bench_with_input(
            BenchmarkId::new("original_serial", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    DirectBuilder::new(rosenbrock, bounds.clone())
                        .algorithm(DirectAlgorithm::GablonskyOriginal)
                        .max_feval(5000)
                        .parallel(false)
                        .minimize()
                        .unwrap()
                })
            },
        );

        // Original — parallel
        group.bench_with_input(
            BenchmarkId::new("original_parallel", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    DirectBuilder::new(rosenbrock, bounds.clone())
                        .algorithm(DirectAlgorithm::GablonskyOriginal)
                        .max_feval(5000)
                        .parallel(true)
                        .minimize()
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_rastrigin(c: &mut Criterion) {
    let mut group = c.benchmark_group("rastrigin");
    group.sample_size(10);

    for &dim in &[2, 5] {
        let bounds = vec![(-5.12, 5.12); dim];

        // Gablonsky — serial
        group.bench_with_input(
            BenchmarkId::new("gablonsky_serial", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    DirectBuilder::new(rastrigin, bounds.clone())
                        .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
                        .max_feval(5000)
                        .parallel(false)
                        .minimize()
                        .unwrap()
                })
            },
        );

        // Gablonsky — parallel
        group.bench_with_input(
            BenchmarkId::new("gablonsky_parallel", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    DirectBuilder::new(rastrigin, bounds.clone())
                        .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
                        .max_feval(5000)
                        .parallel(true)
                        .minimize()
                        .unwrap()
                })
            },
        );

        // Original — serial
        group.bench_with_input(
            BenchmarkId::new("original_serial", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    DirectBuilder::new(rastrigin, bounds.clone())
                        .algorithm(DirectAlgorithm::GablonskyOriginal)
                        .max_feval(5000)
                        .parallel(false)
                        .minimize()
                        .unwrap()
                })
            },
        );

        // Original — parallel
        group.bench_with_input(
            BenchmarkId::new("original_parallel", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    DirectBuilder::new(rastrigin, bounds.clone())
                        .algorithm(DirectAlgorithm::GablonskyOriginal)
                        .max_feval(5000)
                        .parallel(true)
                        .minimize()
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

// ── Parallel threshold tuning benchmarks ──

fn bench_parallel_threshold_cheap(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_threshold_cheap_5d");
    group.sample_size(10);

    let bounds = vec![(-5.0, 5.0); 5];

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

// ── Expensive function benchmarks: parallel speedup at different dimensions ──

fn bench_expensive_rosenbrock(c: &mut Criterion) {
    let mut group = c.benchmark_group("expensive_rosenbrock");
    group.sample_size(10);

    for &dim in &[2, 3, 5, 8] {
        let bounds = vec![(-5.0, 10.0); dim];

        group.bench_with_input(
            BenchmarkId::new("serial", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    DirectBuilder::new(expensive_rosenbrock, bounds.clone())
                        .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
                        .max_feval(500)
                        .parallel(false)
                        .minimize()
                        .unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    DirectBuilder::new(expensive_rosenbrock, bounds.clone())
                        .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
                        .max_feval(500)
                        .parallel(true)
                        .minimize()
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_expensive_rastrigin(c: &mut Criterion) {
    let mut group = c.benchmark_group("expensive_rastrigin");
    group.sample_size(10);

    for &dim in &[2, 3, 5, 8] {
        let bounds = vec![(-5.12, 5.12); dim];

        group.bench_with_input(
            BenchmarkId::new("serial", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    DirectBuilder::new(expensive_rastrigin, bounds.clone())
                        .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
                        .max_feval(500)
                        .parallel(false)
                        .minimize()
                        .unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    DirectBuilder::new(expensive_rastrigin, bounds.clone())
                        .algorithm(DirectAlgorithm::GablonskyLocallyBiased)
                        .max_feval(500)
                        .parallel(true)
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
    bench_sphere,
    bench_rosenbrock,
    bench_rastrigin,
    bench_parallel_threshold_cheap,
    bench_parallel_threshold_expensive,
    bench_expensive_rosenbrock,
    bench_expensive_rastrigin
);
criterion_main!(benches);
