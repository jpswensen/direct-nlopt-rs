# Known Differences Between Rust and NLOPT C Implementations

This document lists all known behavioral differences between the `direct-nlopt-rs`
Rust implementation and the original NLOPT C code.

## Bit-Exact Equivalence (Serial Mode)

With `parallel: false`, the Gablonsky backend (`GablonskyOriginal`,
`GablonskyLocallyBiased`) produces **bit-identical** results to NLOPT's
`GN_ORIG_DIRECT` / `GN_ORIG_DIRECT_L` — same `x`, `fun`, `nfev`, and `nit`.

The CDirect backend (`Original`, `LocallyBiased`, `Randomized`, and their
unscaled variants) produces **bit-identical** results to NLOPT's corresponding
`GN_DIRECT*` algorithms.

## Intentionally Replicated NLOPT Bug

### `dirreplaceinf_` Transposed Array Access (DIRsubrout.c, line 575)

NLOPT's `direct_dirreplaceinf_()` uses `length[rect + dim * n]` to access
sidelengths when checking bounding boxes for infeasible point replacement.
All other functions in NLOPT use `length[dim + rect * n]`. This is a transposed
indexing bug in the original C code.

For `n = 1`, the bug has no effect (transposition is identity). For `n ≥ 2`,
the bug reads incorrect sidelengths, producing wrong bounding boxes for the
`isinbox_` proximity check.

**Our Rust implementation replicates this transposed access** in
`RectangleStorage::replace_infeasible()` to maintain bit-exact compatibility
with NLOPT C. The bug only affects problems with infeasible points (hidden
constraints).

## Differences from NLOPT Behavior

### No Hybrid Mode

NLOPT's `hybrid.c` implements a DIRECT + local optimization hybrid that bisects
(instead of trisects) and runs local optimization within each rectangle. This
mode is **not implemented** in the Rust crate. It corresponds to the NLOPT
algorithms `NLOPT_GN_DIRECT_L_RAND` with local optimization — a rarely used
variant.

### Parallel Evaluation Order

When `parallel: true`, function evaluations occur in a different order than
NLOPT C (which is always sequential). The set of evaluated points is identical,
but the order within a batch may differ due to rayon's work-stealing scheduler.

This affects:
- The `nfev` at which intermediate minimum updates occur (same final result)
- Tie-breaking when multiple points have exactly equal `f`-values (rare)

With `parallel: false`, evaluation order is identical to NLOPT C.

### CDirect Does Not Support Parallel Evaluation

The CDirect backend (SGJ re-implementation) evaluates points sequentially,
matching NLOPT's `cdirect.c`. Only the Gablonsky backend supports the
`parallel` and `parallel_batch` options.

### CDirect Does Not Handle Infeasible Points

NLOPT's `cdirect.c` does not handle `NaN`/`Inf` return values from the
objective function (there is no `dirreplaceinf_` equivalent). If the objective
returns `NaN` in the CDirect backend, behavior is undefined — matching NLOPT.
Use the Gablonsky backend (`GablonskyOriginal`, `GablonskyLocallyBiased`)
for problems with hidden constraints.

### Memory Allocation Strategy

NLOPT's Gablonsky translation pre-allocates fixed arrays sized by
`MAXFUNC` and `MAXDEEP` computed from `maxfeval` and `n`. The Rust
implementation uses the same allocation formula and sizes, but uses
heap-allocated `Vec` instead of C stack/heap arrays. This has no effect on
results but may affect cache behavior and allocation timing.

### Floating-Point Accumulation at High Dimensions

For problems with 5+ dimensions and thousands of evaluations, minor
floating-point accumulation differences between the C and Rust compilers
may produce slightly different results (typically at the 15th+ significant
digit). This is due to different instruction scheduling and FMA usage,
not algorithmic differences. All comparison tests use exact matching for
2D–3D and ≤1e-12 tolerance for higher dimensions.

## Unsupported NLOPT Features

- **Hybrid DIRECT + local optimization** (`hybrid.c`) — Not implemented
- **MPI parallel evaluation** (`DIRparallel.c`) — Not implemented; use rayon instead
- **NLOPT stopping object** — Replaced by `DirectOptions` fields and callback
