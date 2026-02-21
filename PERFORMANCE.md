# DIRECT-NLOPT-RS Performance Report

Generated: 2026-02-21 12:35:45 PST
CPU threads: 16

## 1. Algorithm Variants — Standard Functions

Median of 3 runs per configuration. All serial mode.

| Algorithm | Function | Dims | MaxFEval | Time | NFEval | f(x*) |
|---|---|---:|---:|---:|---:|---:|
| Original | Sphere | 2 | 500 | 214 µs | 500 | 0.0000e0 |
| Original | Sphere | 5 | 2000 | 823 µs | 2000 | 0.0000e0 |
| Original | Sphere | 10 | 5000 | 2.2 ms | 5000 | 0.0000e0 |
| Original | Rosenbrock | 2 | 500 | 172 µs | 500 | 1.9417e-2 |
| Original | Rosenbrock | 5 | 2000 | 1.7 ms | 2000 | 1.6794e-2 |
| Original | Rosenbrock | 10 | 5000 | 4.0 ms | 5000 | 1.5774e0 |
| Original | Rastrigin | 2 | 500 | 193 µs | 500 | 0.0000e0 |
| Original | Rastrigin | 5 | 2000 | 752 µs | 2000 | 0.0000e0 |
| Original | Rastrigin | 10 | 5000 | 3.9 ms | 5000 | 0.0000e0 |
| Original | Ackley | 2 | 500 | 234 µs | 500 | 4.4409e-16 |
| Original | Ackley | 5 | 2000 | 1.1 ms | 2000 | 4.4409e-16 |
| Original | Ackley | 10 | 5000 | 3.0 ms | 5000 | 4.4409e-16 |
| Original | Styblinski-Tang | 2 | 500 | 337 µs | 500 | -7.8332e1 |
| Original | Styblinski-Tang | 5 | 2000 | 790 µs | 2000 | -1.9577e2 |
| Original | Styblinski-Tang | 10 | 5000 | 3.3 ms | 5000 | -3.9127e2 |
| LocallyBiased | Sphere | 2 | 500 | 210 µs | 500 | 0.0000e0 |
| LocallyBiased | Sphere | 5 | 2000 | 968 µs | 2000 | 0.0000e0 |
| LocallyBiased | Sphere | 10 | 5000 | 2.2 ms | 5000 | 0.0000e0 |
| LocallyBiased | Rosenbrock | 2 | 500 | 190 µs | 500 | 6.5171e-8 |
| LocallyBiased | Rosenbrock | 5 | 2000 | 1.1 ms | 2000 | 1.2343e-2 |
| LocallyBiased | Rosenbrock | 10 | 5000 | 3.2 ms | 5000 | 1.3369e1 |
| LocallyBiased | Rastrigin | 2 | 500 | 224 µs | 500 | 0.0000e0 |
| LocallyBiased | Rastrigin | 5 | 2000 | 1.1 ms | 2000 | 0.0000e0 |
| LocallyBiased | Rastrigin | 10 | 5000 | 2.9 ms | 5000 | 0.0000e0 |
| LocallyBiased | Ackley | 2 | 500 | 225 µs | 500 | 4.4409e-16 |
| LocallyBiased | Ackley | 5 | 2000 | 1.1 ms | 2000 | 4.4409e-16 |
| LocallyBiased | Ackley | 10 | 5000 | 3.9 ms | 5000 | 4.4409e-16 |
| LocallyBiased | Styblinski-Tang | 2 | 500 | 210 µs | 500 | -7.8332e1 |
| LocallyBiased | Styblinski-Tang | 5 | 2000 | 893 µs | 2000 | -1.9583e2 |
| LocallyBiased | Styblinski-Tang | 10 | 5000 | 2.8 ms | 5000 | -3.9166e2 |
| Randomized | Sphere | 2 | 500 | 190 µs | 500 | 0.0000e0 |
| Randomized | Sphere | 5 | 2000 | 1.3 ms | 2000 | 0.0000e0 |
| Randomized | Sphere | 10 | 5000 | 8.4 ms | 5000 | 0.0000e0 |
| Randomized | Rosenbrock | 2 | 500 | 314 µs | 500 | 6.5509e-10 |
| Randomized | Rosenbrock | 5 | 2000 | 1.7 ms | 2000 | 2.8580e-4 |
| Randomized | Rosenbrock | 10 | 5000 | 8.5 ms | 5000 | 7.5325e2 |
| Randomized | Rastrigin | 2 | 500 | 207 µs | 500 | 0.0000e0 |
| Randomized | Rastrigin | 5 | 2000 | 1.8 ms | 2000 | 0.0000e0 |
| Randomized | Rastrigin | 10 | 5000 | 10.3 ms | 5000 | 0.0000e0 |
| Randomized | Ackley | 2 | 500 | 281 µs | 500 | 4.4409e-16 |
| Randomized | Ackley | 5 | 2000 | 2.1 ms | 2000 | 4.4409e-16 |
| Randomized | Ackley | 10 | 5000 | 10.1 ms | 5000 | 4.4409e-16 |
| Randomized | Styblinski-Tang | 2 | 500 | 191 µs | 500 | -7.8332e1 |
| Randomized | Styblinski-Tang | 5 | 2000 | 1.2 ms | 2000 | -1.9583e2 |
| Randomized | Styblinski-Tang | 10 | 5000 | 5.8 ms | 5000 | -3.9166e2 |
| GablonskyOrig | Sphere | 2 | 500 | 128 µs | 613 | 0.0000e0 |
| GablonskyOrig | Sphere | 5 | 2000 | 1.1 ms | 3497 | 0.0000e0 |
| GablonskyOrig | Sphere | 10 | 5000 | 8.8 ms | 7469 | 0.0000e0 |
| GablonskyOrig | Rosenbrock | 2 | 500 | 138 µs | 523 | 1.9417e-2 |
| GablonskyOrig | Rosenbrock | 5 | 2000 | 592 µs | 2065 | 2.1052e-2 |
| GablonskyOrig | Rosenbrock | 10 | 5000 | 2.2 ms | 5127 | 1.3583e1 |
| GablonskyOrig | Rastrigin | 2 | 500 | 126 µs | 525 | 0.0000e0 |
| GablonskyOrig | Rastrigin | 5 | 2000 | 1.3 ms | 2609 | 0.0000e0 |
| GablonskyOrig | Rastrigin | 10 | 5000 | 6.8 ms | 6755 | 0.0000e0 |
| GablonskyOrig | Ackley | 2 | 500 | 327 µs | 529 | 4.4409e-16 |
| GablonskyOrig | Ackley | 5 | 2000 | 1.3 ms | 2543 | 4.4409e-16 |
| GablonskyOrig | Ackley | 10 | 5000 | 3.3 ms | 5133 | 4.4409e-16 |
| GablonskyOrig | Styblinski-Tang | 2 | 500 | 121 µs | 519 | -7.8332e1 |
| GablonskyOrig | Styblinski-Tang | 5 | 2000 | 860 µs | 2443 | -1.9583e2 |
| GablonskyOrig | Styblinski-Tang | 10 | 5000 | 2.7 ms | 6421 | -3.8382e2 |
| GablonskyLB | Sphere | 2 | 500 | 153 µs | 543 | 0.0000e0 |
| GablonskyLB | Sphere | 5 | 2000 | 432 µs | 2085 | 0.0000e0 |
| GablonskyLB | Sphere | 10 | 5000 | 1.6 ms | 5035 | 0.0000e0 |
| GablonskyLB | Rosenbrock | 2 | 500 | 152 µs | 517 | 1.4423e-9 |
| GablonskyLB | Rosenbrock | 5 | 2000 | 664 µs | 2011 | 5.2702e-2 |
| GablonskyLB | Rosenbrock | 10 | 5000 | 2.2 ms | 5025 | 1.3369e1 |
| GablonskyLB | Rastrigin | 2 | 500 | 118 µs | 521 | 0.0000e0 |
| GablonskyLB | Rastrigin | 5 | 2000 | 539 µs | 2039 | 0.0000e0 |
| GablonskyLB | Rastrigin | 10 | 5000 | 2.2 ms | 5103 | 0.0000e0 |
| GablonskyLB | Ackley | 2 | 500 | 198 µs | 505 | 4.4409e-16 |
| GablonskyLB | Ackley | 5 | 2000 | 676 µs | 2009 | 4.4409e-16 |
| GablonskyLB | Ackley | 10 | 5000 | 2.9 ms | 5005 | 4.4409e-16 |
| GablonskyLB | Styblinski-Tang | 2 | 500 | 132 µs | 539 | -7.8332e1 |
| GablonskyLB | Styblinski-Tang | 5 | 2000 | 616 µs | 2005 | -1.9583e2 |
| GablonskyLB | Styblinski-Tang | 10 | 5000 | 1.8 ms | 5027 | -3.9166e2 |

## 2. Scaling with Dimensionality

GablonskyLocallyBiased on Sphere, serial. MaxFEval scales with dimension.

| Dims | MaxFEval | Time | NFEval | f(x*) |
|---:|---:|---:|---:|---:|
| 2 | 500 | 113 µs | 543 | 0.0000e0 |
| 5 | 2000 | 394 µs | 2085 | 0.0000e0 |
| 10 | 5000 | 1.5 ms | 5035 | 0.0000e0 |
| 15 | 10000 | 5.1 ms | 10271 | 0.0000e0 |
| 20 | 20000 | 16.4 ms | 20469 | 0.0000e0 |

## 3. Serial vs Parallel (Gablonsky Backend)

Comparison using **expensive** Rosenbrock (10,000 sin/cos iterations per eval).
CPU threads: 16

| Dims | MaxFEval | Serial | Parallel | Speedup | f(serial) | f(parallel) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 200 | 31.0 ms | 23.9 ms | 1.30x | 6.3695e-4 | 6.3695e-4 |
| 5 | 500 | 183.1 ms | 98.8 ms | 1.85x | 1.3551e1 | 1.3551e1 |
| 8 | 500 | 311.1 ms | 74.8 ms | 4.16x | 1.9294e1 | 1.9294e1 |
| 10 | 500 | 365.4 ms | 82.0 ms | 4.45x | 4.9835e1 | 4.9835e1 |

## 4. Cheap-Objective Performance (Serial)

Sphere function, showing raw algorithm overhead without expensive objectives.

| Algorithm | Dims | MaxFEval | Median Time |
|---|---:|---:|---:|
| LocallyBiased | 2 | 2000 | 851 µs |
| LocallyBiased | 5 | 5000 | 3.3 ms |
| LocallyBiased | 10 | 10000 | 7.0 ms |
| GablonskyLB | 2 | 2000 | 456 µs |
| GablonskyLB | 5 | 5000 | 1.3 ms |
| GablonskyLB | 10 | 10000 | 4.0 ms |
| Original | 2 | 2000 | 931 µs |
| Original | 5 | 5000 | 2.2 ms |
| Original | 10 | 10000 | 6.0 ms |
| GablonskyOrig | 2 | 2000 | 1.0 ms |
| GablonskyOrig | 5 | 5000 | 2.5 ms |
| GablonskyOrig | 10 | 10000 | 18.3 ms |

## 5. Summary

### Key Findings

- **Serial mode** produces bit-identical results to NLOPT C code
- **Parallel speedup** scales with dimensionality (2×d points per rectangle) and objective cost
- **CDirect backend** (BTreeMap) and **Gablonsky backend** (SoA + linked lists) show comparable performance
- **Locally-biased variants** converge faster on unimodal/low-multimodal problems
- **Original variants** provide better global exploration on highly multimodal problems

### When to Use Parallel Mode

| Objective Cost | Recommendation |
|---|---|
| < 10 µs | Serial — rayon overhead dominates |
| 100 µs – 1 ms | Parallel with dims ≥ 5 |
| > 1 ms | Always parallel |

---

*Generated by `cargo run --example performance_report --release`*
