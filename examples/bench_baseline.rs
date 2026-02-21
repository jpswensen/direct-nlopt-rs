use std::time::Instant;
use direct_nlopt::{DirectBuilder, types::DirectAlgorithm};

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

fn rosenbrock(x: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..x.len() - 1 {
        let t1 = x[i + 1] - x[i] * x[i];
        let t2 = 1.0 - x[i];
        sum += 100.0 * t1 * t1 + t2 * t2;
    }
    sum
}

fn run_bench(name: &str, dims: usize, f: fn(&[f64]) -> f64, algo: DirectAlgorithm, maxfeval: usize) {
    let bounds: Vec<(f64, f64)> = vec![(-5.0, 5.0); dims];
    let iters = 5;
    let mut times = Vec::new();
    let mut result_fun = 0.0;
    let mut result_nfev = 0;
    for _ in 0..iters {
        let start = Instant::now();
        let result = DirectBuilder::new(f, bounds.clone())
            .algorithm(algo)
            .max_feval(maxfeval)
            .parallel(false)
            .minimize()
            .unwrap();
        times.push(start.elapsed().as_micros());
        result_fun = result.fun;
        result_nfev = result.nfev;
    }
    times.sort();
    let median = times[iters / 2];
    println!("{:<45} {:>8} µs  nfev={:<6} fun={:.6e}", name, median, result_nfev, result_fun);
}

fn main() {
    println!("=== BENCHMARK (serial) ===");
    println!("{:<45} {:>10}  {:<10} {}", "Test", "Time", "NFev", "Fun");
    println!("{}", "-".repeat(85));
    
    run_bench("sphere_2d_gablonsky_lb", 2, sphere, DirectAlgorithm::GablonskyLocallyBiased, 1000);
    run_bench("sphere_5d_gablonsky_lb", 5, sphere, DirectAlgorithm::GablonskyLocallyBiased, 5000);
    run_bench("sphere_10d_gablonsky_lb", 10, sphere, DirectAlgorithm::GablonskyLocallyBiased, 10000);
    run_bench("rosenbrock_2d_gablonsky_lb", 2, rosenbrock, DirectAlgorithm::GablonskyLocallyBiased, 1000);
    run_bench("rosenbrock_5d_gablonsky_lb", 5, rosenbrock, DirectAlgorithm::GablonskyLocallyBiased, 5000);
    run_bench("sphere_2d_gablonsky_orig", 2, sphere, DirectAlgorithm::GablonskyOriginal, 1000);
    run_bench("sphere_5d_gablonsky_orig", 5, sphere, DirectAlgorithm::GablonskyOriginal, 5000);
    run_bench("sphere_10d_gablonsky_orig", 10, sphere, DirectAlgorithm::GablonskyOriginal, 10000);
    run_bench("sphere_2d_cdirect_lb", 2, sphere, DirectAlgorithm::LocallyBiased, 1000);
    run_bench("sphere_5d_cdirect_lb", 5, sphere, DirectAlgorithm::LocallyBiased, 5000);
    run_bench("sphere_10d_cdirect_lb", 10, sphere, DirectAlgorithm::LocallyBiased, 10000);
}
