# direct-nlopt-rs

A faithful Rust-native implementation of the [NLOPT](https://github.com/stevengj/nlopt) DIRECT (DIviding RECTangles) and DIRECT-L global optimization algorithms, with optional rayon parallelization.

## Overview

DIRECT is a deterministic, derivative-free global optimization algorithm for finding the global minimum of a function over a bounded domain. It works by recursively subdividing the search space into hyperrectangles and evaluating the objective at their centers. No gradients, smoothness assumptions, or convexity are required.

This crate ports **both** of NLOPT's DIRECT implementations to Rust:

1. **Gablonsky Fortran→C translation** — Translated from Gablonsky's original Fortran code by Steven G. Johnson. Uses Struct-of-Arrays (SoA) layout with linked lists. Supports DIRECT (Jones 1993) and DIRECT-L (Gablonsky 2001).

2. **SGJ C re-implementation** — From-scratch C implementation by Steven G. Johnson using red-black trees. Supports DIRECT, DIRECT-L, randomized variants, and unscaled coordinate modes.

In serial mode (`parallel: false`), both implementations produce results **bit-identical** to the original NLOPT C code.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
direct-nlopt-rs = "0.1"
```

### Quick Start

```rust
use direct_nlopt::{minimize, DirectBuilder, DirectAlgorithm};

// Simple: minimize sphere function with DIRECT-L (default)
let result = minimize(
    |x: &[f64]| x.iter().map(|xi| xi * xi).sum(),
    &vec![(-5.0, 5.0), (-5.0, 5.0)],
    500,
).unwrap();
println!("Minimum: {:.6} at {:?}", result.fun, result.x);
```

### Builder Pattern

```rust
use direct_nlopt::{DirectBuilder, DirectAlgorithm};

let result = DirectBuilder::new(
    |x: &[f64]| {
        // Rosenbrock function
        100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2)
    },
    vec![(-5.0, 5.0), (-5.0, 5.0)],
)
.algorithm(DirectAlgorithm::GablonskyLocallyBiased)
.max_feval(5000)
.parallel(true)
.minimize()
.unwrap();

assert!(result.fun < 0.01);
```

### Full Configuration

```rust
use direct_nlopt::{direct_optimize, DirectOptions, DirectAlgorithm, DIRECT_UNKNOWN_FGLOBAL};

let result = direct_optimize(
    |x: &[f64]| x.iter().map(|xi| xi * xi).sum(),
    &vec![(-5.0, 5.0); 3],
    DirectOptions {
        max_feval: 5000,
        max_iter: 200,
        max_time: 30.0,
        magic_eps: 1e-4,
        magic_eps_abs: 0.0,
        volume_reltol: 0.0,
        sigma_reltol: -1.0,
        fglobal: DIRECT_UNKNOWN_FGLOBAL,
        fglobal_reltol: 0.0,
        algorithm: DirectAlgorithm::GablonskyLocallyBiased,
        parallel: true,
        parallel_batch: false,
        min_parallel_evals: 4,
    },
).unwrap();
```

### Callback for Progress Monitoring / Early Stopping

```rust
use direct_nlopt::{DirectBuilder, DirectAlgorithm};

let result = DirectBuilder::new(
    |x: &[f64]| x.iter().map(|xi| xi * xi).sum(),
    vec![(-5.0, 5.0), (-5.0, 5.0)],
)
.max_feval(5000)
.algorithm(DirectAlgorithm::LocallyBiased)
.with_callback(|x_best, f_best, nfev, nit| {
    println!("Iteration {nit}: f={f_best:.6e}, nfev={nfev}");
    f_best < 1e-6  // return true to stop early
})
.minimize()
.unwrap();
```

### Hidden Constraints (Infeasible Regions)

Return `f64::NAN` or `f64::INFINITY` to mark points as infeasible:

```rust
use direct_nlopt::{DirectBuilder, DirectAlgorithm};

let result = DirectBuilder::new(
    |x: &[f64]| {
        // Infeasible outside unit circle
        if x[0] * x[0] + x[1] * x[1] > 1.0 {
            return f64::NAN;
        }
        (x[0] - 0.3).powi(2) + (x[1] - 0.4).powi(2)
    },
    vec![(-2.0, 2.0), (-2.0, 2.0)],
)
.algorithm(DirectAlgorithm::GablonskyLocallyBiased)
.max_feval(1000)
.minimize()
.unwrap();
```

## Algorithm Variants

| Rust Variant | NLOPT Name | Backend | Description |
|---|---|---|---|
| `DirectAlgorithm::Original` | `GN_DIRECT` | CDirect | Jones' original DIRECT (1993) |
| `DirectAlgorithm::LocallyBiased` | `GN_DIRECT_L` | CDirect | DIRECT-L, locally biased (**default**) |
| `DirectAlgorithm::Randomized` | `GN_DIRECT_L_RAND` | CDirect | DIRECT-L with randomized tie-breaking |
| `DirectAlgorithm::OriginalUnscaled` | `GN_DIRECT_NOSCAL` | CDirect | Original, unscaled coordinates |
| `DirectAlgorithm::LocallyBiasedUnscaled` | `GN_DIRECT_L_NOSCAL` | CDirect | DIRECT-L, unscaled |
| `DirectAlgorithm::LocallyBiasedRandomizedUnscaled` | `GN_DIRECT_L_RAND_NOSCAL` | CDirect | DIRECT-L randomized, unscaled |
| `DirectAlgorithm::GablonskyOriginal` | `GN_ORIG_DIRECT` | Gablonsky | Jones' original via Fortran translation |
| `DirectAlgorithm::GablonskyLocallyBiased` | `GN_ORIG_DIRECT_L` | Gablonsky | DIRECT-L via Fortran translation |

### Which Variant to Choose?

- **`LocallyBiased` (default)**: Good general-purpose choice. Uses the SGJ cdirect backend.
- **`GablonskyLocallyBiased`**: Equivalent algorithm via the Gablonsky backend. Supports parallelization, hidden constraints (infeasible regions), and produces identical results to NLOPT's `GN_ORIG_DIRECT_L`.
- **`Original`**: Jones' original DIRECT. Less locally biased — better for highly multimodal functions but slower to converge near optima.
- **`Randomized`**: Breaks ties randomly. Can help escape symmetry traps.

## NLOPT Correspondence

### Function Mapping — Gablonsky Translation

| NLOPT C Function | Rust Equivalent | Module |
|---|---|---|
| `direct_direct_()` | `Direct::minimize()` | `direct.rs` |
| `direct_dirinit_()` | `Direct::initialize()` | `direct.rs` |
| `direct_dirpreprc_()` | `Direct::new()` scaling | `direct.rs` |
| `direct_dirheader_()` | `Direct::validate_inputs()` | `direct.rs` |
| `direct_dirsamplepoints_()` | `Direct::sample_points()` | `direct.rs` |
| `direct_dirsamplef_()` | `Direct::evaluate_sample_points()` | `direct.rs` |
| `direct_dirdivide_()` | `Direct::divide_rectangle()` | `direct.rs` |
| `direct_dirchoose_()` | `PotentiallyOptimal::select()` | `storage.rs` |
| `direct_dirdoubleinsert_()` | `PotentiallyOptimal::double_insert()` | `storage.rs` |
| `direct_dirinsertlist_()` | `RectangleStorage::insert_into_list()` | `storage.rs` |
| `direct_dirget_i__()` | `RectangleStorage::get_longest_dims()` | `storage.rs` |
| `direct_dirgetlevel_()` | `RectangleStorage::get_level()` | `storage.rs` |
| `direct_dirgetmaxdeep_()` | `RectangleStorage::get_max_deep()` | `storage.rs` |
| `direct_dirreplaceinf_()` | `RectangleStorage::replace_infeasible()` | `storage.rs` |
| `direct_dirinfcn_()` | `Direct::evaluate()` + `to_actual()` | `direct.rs` |
| `direct_optimize()` | `direct_optimize()` / `DirectBuilder` | `lib.rs` |

### Function Mapping — SGJ CDirect

| NLOPT C Function | Rust Equivalent | Module |
|---|---|---|
| `cdirect_unscaled()` | `CDirect::optimize_unscaled()` | `cdirect.rs` |
| `cdirect()` | `CDirect::optimize()` | `cdirect.rs` |
| `divide_rect()` | `CDirect::divide_rect()` | `cdirect.rs` |
| `convex_hull()` | `CDirect::convex_hull()` | `cdirect.rs` |
| `divide_good_rects()` | `CDirect::divide_good_rects()` | `cdirect.rs` |
| `rect_diameter()` | `CDirect::rect_diameter()` | `cdirect.rs` |

### Data Structure Mapping — Gablonsky Backend

| Data | NLOPT C Variable | Rust Field (`RectangleStorage`) |
|---|---|---|
| Centers | `c__[MAXFUNC×n]` | `centers: Vec<f64>` |
| F-values | `f[MAXFUNC×2]` | `f_values: Vec<f64>` |
| Side lengths | `length[MAXFUNC×n]` | `lengths: Vec<i32>` |
| Linked list | `point[MAXFUNC]` | `point: Vec<i32>` |
| Depth anchors | `anchor[MAXDEEP+2]` | `anchor: Vec<i32>` |
| Free list head | `free` | `free: i32` |
| Precomp thirds | `thirds[MAXDEEP+1]` | `thirds: Vec<f64>` |
| Precomp levels | `levels[MAXDEEP+1]` | `levels: Vec<f64>` |

## Parallelization

Function evaluations can be parallelized using [rayon](https://docs.rs/rayon). Parallelization is available for the Gablonsky backend (`GablonskyOriginal`, `GablonskyLocallyBiased`).

### Options

| Option | Default | Description |
|---|---|---|
| `parallel` | `false` | Enable per-rectangle parallel evaluation of 2×d sample points |
| `parallel_batch` | `false` | Batch evaluations across all selected rectangles per iteration |
| `min_parallel_evals` | `4` | Minimum batch size to trigger parallel execution |

### When to Use Parallel Mode

- **Cheap objectives** (microseconds): Keep `parallel: false`. Rayon overhead (~1–5 µs per spawn) dominates.
- **Moderate objectives** (milliseconds): Use `parallel: true`. Speedup scales with dimensionality.
- **Expensive objectives** (seconds): Use `parallel: true` and `parallel_batch: true`. Maximum throughput.
- **1D/2D problems**: Only 2–4 sample points per rectangle — serial is faster regardless.

Control rayon thread count via `RAYON_NUM_THREADS` environment variable.

### Correctness Guarantee

With `parallel: false`, evaluation order is identical to NLOPT C, producing **bit-identical** results. With `parallel: true`, the same points are evaluated but in a different order; final results are equivalent.

## Performance

Rust serial performance is approximately 0.5–0.7× NLOPT C at `-O3` for cheap objective functions. The overhead comes from Rust's additional safety checks, dynamic dispatch through `Box<dyn Fn>`, and `Vec` allocation vs C's stack-allocated arrays.

For expensive objective functions, parallel mode provides significant speedup that scales with dimensionality and objective cost.

## C FFI

The crate exposes a C-compatible FFI matching NLOPT's `direct_optimize()` signature. Enable the `ffi` feature and use the generated `direct_nlopt.h` header. See `src/ffi.rs` for details.

## Feature Flags

| Feature | Description |
|---|---|
| `trace` | Enable step-by-step algorithm tracing for debugging |
| `nlopt-compare` | Compile NLOPT C code for comparison testing (requires C compiler) |
| `ffi` | C FFI bindings |

## Termination Criteria

| Option | Code | Description |
|---|---|---|
| `max_feval` | `MaxFevalExceeded` | Maximum function evaluations reached |
| `max_iter` | `MaxIterExceeded` | Maximum iterations reached |
| `max_time` | `MaxTimeExceeded` | Wall-clock time limit reached |
| `fglobal` + `fglobal_reltol` | `GlobalFound` | `f ≤ fglobal + reltol × |fglobal|` |
| `volume_reltol` | `VolTol` | Smallest rectangle volume below threshold |
| `sigma_reltol` | `SigmaTol` | Rectangle measure below threshold |
| Callback returns `true` | `ForcedStop` | User-defined early stopping |

## References

- Jones, D.R., Perttunen, C.D. & Stuckman, B.E. "Lipschitzian optimization without the Lipschitz constant." *J Optim Theory Appl* 79, 157–181 (1993).
- Gablonsky, J.M. & Kelley, C.T. "A Locally-Biased form of the DIRECT Algorithm." *Journal of Global Optimization* 21, 27–37 (2001).
- NLOPT: <https://github.com/stevengj/nlopt>

## Attribution

The bulk of this implementation was generated using **Claude Opus 4.6** (Anthropic) and the
**ralph-wiggum method** — an automated multi-step agentic workflow that iteratively builds,
tests, and refines code through a structured PRD (Product Requirements Document) plan. The
NLOPT C source code (both the Gablonsky Fortran→C translation and the SGJ cdirect
re-implementation) served as the sole source material for the faithful port.

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for
details. This matches the license of the NLOPT DIRECT source code from which this
implementation was derived.
