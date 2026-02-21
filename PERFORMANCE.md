# DIRECT-NLOPT-RS Performance Report

Generated: 2026-02-21 13:06:03 PST
CPU threads: 16

## 1. Algorithm Variants — Standard Functions

Median of 3 runs per configuration. All serial mode.

| Algorithm | Function | Dims | MaxFEval | Time | NFEval | f(x*) |
|---|---|---:|---:|---:|---:|---:|
| Original | Sphere | 2 | 500 | 260 µs | 500 | 0.0000e0 |
| Original | Sphere | 5 | 2000 | 1.1 ms | 2000 | 0.0000e0 |
| Original | Sphere | 10 | 5000 | 4.0 ms | 5000 | 0.0000e0 |
| Original | Rosenbrock | 2 | 500 | 283 µs | 500 | 1.9417e-2 |
| Original | Rosenbrock | 5 | 2000 | 1.2 ms | 2000 | 1.6794e-2 |
| Original | Rosenbrock | 10 | 5000 | 4.1 ms | 5000 | 1.5774e0 |
| Original | Rastrigin | 2 | 500 | 261 µs | 500 | 0.0000e0 |
| Original | Rastrigin | 5 | 2000 | 910 µs | 2000 | 0.0000e0 |
| Original | Rastrigin | 10 | 5000 | 4.1 ms | 5000 | 0.0000e0 |
| Original | Ackley | 2 | 500 | 331 µs | 500 | 4.4409e-16 |
| Original | Ackley | 5 | 2000 | 1.2 ms | 2000 | 4.4409e-16 |
| Original | Ackley | 10 | 5000 | 3.8 ms | 5000 | 4.4409e-16 |
| Original | Styblinski-Tang | 2 | 500 | 257 µs | 500 | -7.8332e1 |
| Original | Styblinski-Tang | 5 | 2000 | 1.0 ms | 2000 | -1.9577e2 |
| Original | Styblinski-Tang | 10 | 5000 | 3.1 ms | 5000 | -3.9127e2 |
| LocallyBiased | Sphere | 2 | 500 | 281 µs | 500 | 0.0000e0 |
| LocallyBiased | Sphere | 5 | 2000 | 1.1 ms | 2000 | 0.0000e0 |
| LocallyBiased | Sphere | 10 | 5000 | 3.1 ms | 5000 | 0.0000e0 |
| LocallyBiased | Rosenbrock | 2 | 500 | 298 µs | 500 | 6.5171e-8 |
| LocallyBiased | Rosenbrock | 5 | 2000 | 1.5 ms | 2000 | 1.2343e-2 |
| LocallyBiased | Rosenbrock | 10 | 5000 | 4.2 ms | 5000 | 1.3369e1 |
| LocallyBiased | Rastrigin | 2 | 500 | 285 µs | 500 | 0.0000e0 |
| LocallyBiased | Rastrigin | 5 | 2000 | 1.2 ms | 2000 | 0.0000e0 |
| LocallyBiased | Rastrigin | 10 | 5000 | 3.6 ms | 5000 | 0.0000e0 |
| LocallyBiased | Ackley | 2 | 500 | 332 µs | 500 | 4.4409e-16 |
| LocallyBiased | Ackley | 5 | 2000 | 1.5 ms | 2000 | 4.4409e-16 |
| LocallyBiased | Ackley | 10 | 5000 | 3.5 ms | 5000 | 4.4409e-16 |
| LocallyBiased | Styblinski-Tang | 2 | 500 | 276 µs | 500 | -7.8332e1 |
| LocallyBiased | Styblinski-Tang | 5 | 2000 | 1.2 ms | 2000 | -1.9583e2 |
| LocallyBiased | Styblinski-Tang | 10 | 5000 | 3.5 ms | 5000 | -3.9166e2 |
| Randomized | Sphere | 2 | 500 | 266 µs | 500 | 0.0000e0 |
| Randomized | Sphere | 5 | 2000 | 1.7 ms | 2000 | 0.0000e0 |
| Randomized | Sphere | 10 | 5000 | 11.8 ms | 5000 | 0.0000e0 |
| Randomized | Rosenbrock | 2 | 500 | 292 µs | 500 | 6.5509e-10 |
| Randomized | Rosenbrock | 5 | 2000 | 1.7 ms | 2000 | 2.8580e-4 |
| Randomized | Rosenbrock | 10 | 5000 | 8.6 ms | 5000 | 7.5325e2 |
| Randomized | Rastrigin | 2 | 500 | 300 µs | 500 | 0.0000e0 |
| Randomized | Rastrigin | 5 | 2000 | 1.8 ms | 2000 | 0.0000e0 |
| Randomized | Rastrigin | 10 | 5000 | 12.5 ms | 5000 | 0.0000e0 |
| Randomized | Ackley | 2 | 500 | 331 µs | 500 | 4.4409e-16 |
| Randomized | Ackley | 5 | 2000 | 2.7 ms | 2000 | 4.4409e-16 |
| Randomized | Ackley | 10 | 5000 | 11.7 ms | 5000 | 4.4409e-16 |
| Randomized | Styblinski-Tang | 2 | 500 | 266 µs | 500 | -7.8332e1 |
| Randomized | Styblinski-Tang | 5 | 2000 | 1.5 ms | 2000 | -1.9583e2 |
| Randomized | Styblinski-Tang | 10 | 5000 | 6.6 ms | 5000 | -3.9166e2 |
| GablonskyOrig | Sphere | 2 | 500 | 200 µs | 613 | 0.0000e0 |
| GablonskyOrig | Sphere | 5 | 2000 | 1.4 ms | 3497 | 0.0000e0 |
| GablonskyOrig | Sphere | 10 | 5000 | 7.2 ms | 7469 | 0.0000e0 |
| GablonskyOrig | Rosenbrock | 2 | 500 | 229 µs | 523 | 1.9417e-2 |
| GablonskyOrig | Rosenbrock | 5 | 2000 | 712 µs | 2065 | 2.1052e-2 |
| GablonskyOrig | Rosenbrock | 10 | 5000 | 2.0 ms | 5127 | 1.3583e1 |
| GablonskyOrig | Rastrigin | 2 | 500 | 176 µs | 525 | 0.0000e0 |
| GablonskyOrig | Rastrigin | 5 | 2000 | 1.4 ms | 2609 | 0.0000e0 |
| GablonskyOrig | Rastrigin | 10 | 5000 | 6.7 ms | 6755 | 0.0000e0 |
| GablonskyOrig | Ackley | 2 | 500 | 209 µs | 529 | 4.4409e-16 |
| GablonskyOrig | Ackley | 5 | 2000 | 1.2 ms | 2543 | 4.4409e-16 |
| GablonskyOrig | Ackley | 10 | 5000 | 3.5 ms | 5133 | 4.4409e-16 |
| GablonskyOrig | Styblinski-Tang | 2 | 500 | 188 µs | 519 | -7.8332e1 |
| GablonskyOrig | Styblinski-Tang | 5 | 2000 | 1.1 ms | 2443 | -1.9583e2 |
| GablonskyOrig | Styblinski-Tang | 10 | 5000 | 3.7 ms | 6421 | -3.8382e2 |
| GablonskyLB | Sphere | 2 | 500 | 189 µs | 543 | 0.0000e0 |
| GablonskyLB | Sphere | 5 | 2000 | 559 µs | 2085 | 0.0000e0 |
| GablonskyLB | Sphere | 10 | 5000 | 1.5 ms | 5035 | 0.0000e0 |
| GablonskyLB | Rosenbrock | 2 | 500 | 192 µs | 517 | 1.4423e-9 |
| GablonskyLB | Rosenbrock | 5 | 2000 | 794 µs | 2011 | 5.2702e-2 |
| GablonskyLB | Rosenbrock | 10 | 5000 | 2.4 ms | 5025 | 1.3369e1 |
| GablonskyLB | Rastrigin | 2 | 500 | 187 µs | 521 | 0.0000e0 |
| GablonskyLB | Rastrigin | 5 | 2000 | 722 µs | 2039 | 0.0000e0 |
| GablonskyLB | Rastrigin | 10 | 5000 | 2.1 ms | 5103 | 0.0000e0 |
| GablonskyLB | Ackley | 2 | 500 | 252 µs | 505 | 4.4409e-16 |
| GablonskyLB | Ackley | 5 | 2000 | 829 µs | 2009 | 4.4409e-16 |
| GablonskyLB | Ackley | 10 | 5000 | 2.4 ms | 5005 | 4.4409e-16 |
| GablonskyLB | Styblinski-Tang | 2 | 500 | 181 µs | 539 | -7.8332e1 |
| GablonskyLB | Styblinski-Tang | 5 | 2000 | 724 µs | 2005 | -1.9583e2 |
| GablonskyLB | Styblinski-Tang | 10 | 5000 | 2.2 ms | 5027 | -3.9166e2 |

## 2. Scaling with Dimensionality

GablonskyLocallyBiased on Sphere, serial. MaxFEval scales with dimension.

| Dims | MaxFEval | Time | NFEval | f(x*) |
|---:|---:|---:|---:|---:|
| 2 | 500 | 171 µs | 543 | 0.0000e0 |
| 5 | 2000 | 567 µs | 2085 | 0.0000e0 |
| 10 | 5000 | 1.5 ms | 5035 | 0.0000e0 |
| 15 | 10000 | 6.0 ms | 10271 | 0.0000e0 |
| 20 | 20000 | 17.7 ms | 20469 | 0.0000e0 |

## 3. Serial vs Parallel (Gablonsky Backend)

Comparison using **expensive** Rosenbrock (10,000 sin/cos iterations per eval).
CPU threads: 16

| Dims | MaxFEval | Serial | Parallel | Speedup | f(serial) | f(parallel) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 200 | 33.7 ms | 24.4 ms | 1.38x | 6.3695e-4 | 6.3695e-4 |
| 5 | 500 | 315.9 ms | 96.1 ms | 3.29x | 1.3551e1 | 1.3551e1 |
| 8 | 500 | 375.0 ms | 90.6 ms | 4.14x | 1.9294e1 | 1.9294e1 |
| 10 | 500 | 456.0 ms | 97.5 ms | 4.68x | 4.9835e1 | 4.9835e1 |

## 3b. Serial vs Parallel (CDirect Backend)

Comparison using **expensive** Rosenbrock (10,000 sin/cos iterations per eval).
CPU threads: 16

| Algorithm | Dims | MaxFEval | Serial | Parallel | Speedup | f(serial) | f(parallel) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 2 | 200 | 30.1 ms | 8.5 ms | 3.52x | 2.7462e-2 | 2.7024e-2 |
| Original | 5 | 500 | 198.3 ms | 38.1 ms | 5.21x | 3.2068e0 | 2.1432e0 |
| Original | 8 | 500 | 361.5 ms | 80.0 ms | 4.52x | 3.1792e1 | 3.1792e1 |
| Original | 10 | 500 | 448.2 ms | 88.8 ms | 5.05x | 6.9603e1 | 6.9603e1 |
| LocallyBiased | 2 | 200 | 31.0 ms | 9.1 ms | 3.40x | 2.0691e-2 | 2.0691e-2 |
| LocallyBiased | 5 | 500 | 204.8 ms | 47.1 ms | 4.35x | 1.3551e1 | 1.3551e1 |
| LocallyBiased | 8 | 500 | 352.5 ms | 70.8 ms | 4.98x | 1.9485e1 | 1.9485e1 |
| LocallyBiased | 10 | 500 | 425.8 ms | 93.6 ms | 4.55x | 4.9835e1 | 4.9835e1 |
| Randomized | 2 | 200 | 34.5 ms | 11.7 ms | 2.95x | 9.6560e-6 | 2.1509e-2 |
| Randomized | 5 | 500 | 213.9 ms | 48.1 ms | 4.45x | 1.8376e1 | 1.7607e1 |
| Randomized | 8 | 500 | 369.7 ms | 99.9 ms | 3.70x | 5.0574e2 | 5.0574e2 |
| Randomized | 10 | 500 | 454.3 ms | 119.8 ms | 3.79x | 7.5452e2 | 7.5452e2 |

## 4. Cheap-Objective Performance (Serial)

Sphere function, showing raw algorithm overhead without expensive objectives.

| Algorithm | Dims | MaxFEval | Median Time |
|---|---:|---:|---:|
| LocallyBiased | 2 | 2000 | 1.1 ms |
| LocallyBiased | 5 | 5000 | 3.3 ms |
| LocallyBiased | 10 | 10000 | 9.8 ms |
| GablonskyLB | 2 | 2000 | 752 µs |
| GablonskyLB | 5 | 5000 | 1.9 ms |
| GablonskyLB | 10 | 10000 | 4.5 ms |
| Original | 2 | 2000 | 1.0 ms |
| Original | 5 | 5000 | 2.7 ms |
| Original | 10 | 10000 | 8.0 ms |
| GablonskyOrig | 2 | 2000 | 996 µs |
| GablonskyOrig | 5 | 5000 | 2.9 ms |
| GablonskyOrig | 10 | 10000 | 22.0 ms |

## 5. Summary

### Key Findings

- **Serial mode** produces bit-identical results to NLOPT C code
- **Parallel speedup** scales with dimensionality and objective cost for both backends
- **CDirect parallel** uses collect→parallel-eval→apply across all potentially-optimal rectangles
- **Gablonsky parallel** uses per-rectangle and batch-across-rectangle parallel evaluation
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
