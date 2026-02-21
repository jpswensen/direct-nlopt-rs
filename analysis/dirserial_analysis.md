# Analysis of DIRserial.c: Serial Function Evaluation

## Overview

`DIRserial.c` contains a single function: `direct_dirsamplef_()`, the **serial version** of function
evaluation for the DIRECT algorithm. This file is the counterpart to `DIRparallel.c` which implements
a PVM-based parallel version. In NLOPT's build, the serial version is always compiled (the parallel
version requires PVM which is not used in NLOPT).

This function is the **primary target for rayon-based parallelization** in the Rust port.

**File**: `nlopt/src/algs/direct/DIRserial.c` (150 lines)
**Original**: Translated from Gablonsky's Fortran code `DIRserial.f` by f2c, hand-cleaned by SGJ (August 2007).

---

## Function: `direct_dirsamplef_()`

### Signature

```c
void direct_dirsamplef_(
    doublereal *c__,          // Centers array [n × MAXFUNC], row-major (SGJ transposed)
    integer *arrayi,          // Array of dimension indices to divide (1-based)
    doublereal *delta,        // UNUSED in serial version (only referenced via `sample` in parallel)
    integer *sample,          // UNUSED in serial version
    integer *new__,           // Index of the first new sample point (1-based)
    integer *length,          // Side-length indices array [n × MAXFUNC]
    FILE *logfile,            // UNUSED in serial version
    doublereal *f,            // Function values array [2 × MAXFUNC]: f[2*pos]=value, f[2*pos+1]=flag
    integer *free,            // UNUSED in serial version (free list head)
    integer *maxi,            // Number of longest dimensions being divided
    integer *point,           // Linked list pointers [MAXFUNC]
    fp fcn,                   // User objective function
    doublereal *x,            // Scratch buffer for coordinates [n]
    doublereal *l,            // Lower bounds (actually xs1 = u-l after dirpreprc_)
    doublereal *minf,         // IN/OUT: current global minimum function value
    integer *minpos,          // IN/OUT: position of global minimum
    doublereal *u,            // Upper bounds (actually xs2 = l/(u-l) after dirpreprc_)
    integer *n,               // Problem dimension
    integer *maxfunc,         // MAXFUNC capacity
    const integer *maxdeep,   // UNUSED in serial version
    integer *oops,            // UNUSED in serial version (error flag)
    doublereal *fmax,         // IN/OUT: maximum feasible function value seen so far
    integer *ifeasiblef,      // IN/OUT: 0 if any feasible point found, 1 otherwise
    integer *iinfesiblef,     // IN/OUT: max infeasibility flag seen (0=feasible, ≥1=infeasible, -1=setup error)
    void *fcn_data,           // Opaque user data passed to fcn
    int *force_stop           // Pointer to force-stop flag (set externally to abort)
)
```

### Unused Parameters

Several parameters are explicitly cast to `(void)` at the top of the function:
- `logfile`, `free`, `maxfunc`, `maxdeep`, `oops`, `delta`, `sample`

These are part of the shared signature with `DIRparallel.c` but are not needed in the serial version.

### Parameter Adjustments (f2c 1-Based Indexing)

The function applies Fortran-to-C index adjustments:
```c
--u;                    // u becomes 1-based: u[1..n]
--l;                    // l becomes 1-based: l[1..n]
--x;                    // x becomes 1-based: x[1..n]
--arrayi;               // arrayi becomes 1-based: arrayi[1..maxi]
--point;                // point becomes 1-based: point[1..MAXFUNC]
f -= 3;                 // f becomes 1-based with stride 2: f[2*pos+1]=value, f[2*pos+2]=flag
```

The `f` adjustment `f -= 3` is key. After adjustment:
- `f[(pos << 1) + 1]` = f-value at position `pos` (the objective function result)
- `f[(pos << 1) + 2]` = feasibility flag at position `pos` (0=feasible, 2=infeasible, -1=setup error)

The `c__` and `length` adjustments use dimension-based offsets:
```c
c_dim1 = *n;
c_offset = 1 + c_dim1;
c__ -= c_offset;        // c__[i + pos * c_dim1] for 1-based i, pos

length_dim1 = *n;
length_offset = 1 + length_dim1;
length -= length_offset; // length[i + pos * length_dim1] (unused in serial but adjusted)
```

Note: `c__` and `length` adjustments are consistent with SGJ's row-major transposition where
the first index is the dimension (1..n) and the second is the rectangle position.

---

## Algorithm: Two-Pass Structure

### Pass 1: Evaluate Function at All 2×maxi New Points (lines 67–133)

```
pos = new__               // Start at first new sample point
helppoint = pos            // Save start position for Pass 2

FOR j = 1 TO 2*maxi:      // Iterate over all new sample points
    // Copy center coordinates of rectangle at pos into scratch buffer x
    FOR i = 1 TO n:
        x[i] = c__[i + pos * n]   // Row-major: dim i, rect pos

    // Evaluate objective function (or skip if force_stop)
    IF force_stop AND *force_stop:
        f[2*pos+1] = fmax         // Assign fmax as dummy value
    ELSE:
        direct_dirinfcn_(fcn, x, l, u, n, &f[2*pos+1], &kret, fcn_data)
        // dirinfcn_ unscales x: x_actual = (x + xs2) * xs1
        // calls fcn(n, x_actual, &kret, fcn_data)
        // rescales x back: x = x_actual / xs1 - xs2

    IF force_stop AND *force_stop:
        kret = -1                 // Mark as invalid point

    // Update global infeasibility tracking
    iinfesiblef = MAX(iinfesiblef, kret)

    // Handle evaluation result based on kret
    IF kret == 0:                 // Feasible evaluation succeeded
        f[2*pos+2] = 0.0         // Flag: feasible
        ifeasiblef = 0            // At least one feasible point exists
        fmax = MAX(f[2*pos+1], fmax)  // Track max feasible value

    IF kret >= 1:                 // Infeasible point
        f[2*pos+2] = 2.0         // Flag: infeasible
        f[2*pos+1] = fmax        // Replace f-value with current max

    IF kret == -1:                // Setup failure
        f[2*pos+2] = -1.0        // Flag: setup error

    // Advance to next sample point via linked list
    pos = point[pos]
```

### Pass 2: Update Global Minimum (lines 134–149)

```
pos = helppoint            // Reset to first new sample point

FOR j = 1 TO 2*maxi:
    // Check if this point has a new minimum (only feasible points)
    IF f[2*pos+1] < minf AND f[2*pos+2] == 0.0:
        minf = f[2*pos+1]
        minpos = pos

    // Advance to next sample point
    pos = point[pos]
```

---

## Evaluation Order (Critical for Faithfulness)

The evaluation order follows the **linked list traversal** starting from `new__`:

```
new__ → point[new__] → point[point[new__]] → ... (2*maxi steps)
```

The linked list is constructed by `direct_dirsamplepoints_()` which creates points in this order:
1. For dimension `arrayi[1]`: positive offset point, then negative offset point
2. For dimension `arrayi[2]`: positive offset point, then negative offset point
3. ...
4. For dimension `arrayi[maxi]`: positive offset point, then negative offset point

So for `maxi=3` with dimensions `arrayi = [d1, d2, d3]`, the evaluation order is:
```
center+delta_d1, center-delta_d1, center+delta_d2, center-delta_d2, center+delta_d3, center-delta_d3
```

**This is exactly `2*maxi` evaluations**, one for each new sample point.

The linked list chain is: `new__ → pos1_neg → pos2_pos → pos2_neg → pos3_pos → pos3_neg`
(where the initial `new__` is `pos1_pos`).

### Verification of Evaluation Order from dirsamplepoints_()

Looking at `direct_dirsamplepoints_()` in DIRsubrout.c (lines ~1130-1185):
```
FOR j = 1 TO maxi:
    // Positive offset: copy center, then x[arrayi[j]] += delta
    // Allocate from free list → pos_plus = free; free = point[free]
    // Link: point[sample_prev] = pos_plus
    
    // Negative offset: copy center, then x[arrayi[j]] -= delta
    // Allocate from free list → pos_minus = free; free = point[free]
    // Link: point[pos_plus] = pos_minus
    
    // Save: sample_prev = pos_minus for next iteration
```

So the linked list chains as: `start → plus1 → minus1 → plus2 → minus2 → ... → plusM → minusM`

**Wait** — re-reading the C code more carefully:

In `dirsamplepoints_()`, for each dimension j:
- First new point (line ~1150): `c[i + new * n] = center ± delta` — this is the POSITIVE offset
  - `point[sample] = new; sample = new; new = point[new]` — chains it
- Second new point (line ~1170): same but NEGATIVE offset
  - `point[sample] = new; sample = new; new = point[new]` — chains it

So the chain is: `start → pos_1 → neg_1 → pos_2 → neg_2 → ... → pos_M → neg_M`

And `dirsamplef_` evaluates in this exact order by following `pos = point[pos]`.

---

## What is Independent vs Sequential

### Independent Operations (parallelizable)

Each of the `2*maxi` function evaluations is **completely independent**:
- Each point has its own center coordinates stored in `c__[*, pos]`
- Each result is stored at a unique position `f[2*pos+1]`, `f[2*pos+2]`
- The `dirinfcn_()` call only reads `l` (xs1) and `u` (xs2) scaling coefficients (shared, read-only)
- The user function `fcn` is expected to be thread-safe (const access to fcn_data)

The scratch buffer `x[1..n]` is the **only shared mutable state** — each evaluation needs its own copy.

### Sequential Operations (must remain ordered)

1. **fmax tracking**: `fmax = MAX(fmax, f_value)` — accumulates across evaluations.
   Can be computed after all evaluations complete.

2. **iinfesiblef tracking**: `iinfesiblef = MAX(iinfesiblef, kret)` — accumulates.
   Can be computed after all evaluations complete.

3. **ifeasiblef tracking**: set to 0 if any feasible point found.
   Can be computed after all evaluations complete.

4. **minf/minpos update** (Pass 2): scans all evaluated points for minimum.
   This is already a separate pass and can be trivially parallelized or done after.

5. **Infeasible value assignment**: `f[2*pos+1] = fmax` for infeasible points uses the
   *current* fmax at the time of evaluation. In the serial version, this means infeasible
   points evaluated later might get a different fmax than those evaluated earlier.
   **However**, this is a minor detail — in practice, fmax changes rarely within a single
   batch of 2*maxi evaluations.

### Key Insight for Parallelization

The ONLY truly order-dependent behavior is the infeasible point value assignment:
```c
if (kret >= 1) {
    f[(pos << 1) + 2] = 2.;
    f[(pos << 1) + 1] = *fmax;    // Uses fmax accumulated up to this point
}
```

In parallel mode, all evaluations happen simultaneously, so `fmax` would be the value from
BEFORE this batch. This means infeasible points in the parallel version might get a slightly
different replacement value than the serial version. This is acceptable because:
1. Infeasible point values are later replaced by `dirreplaceinf_()` anyway
2. The replacement value is just a placeholder (max feasible value)
3. The final optimization result is unaffected

**For exact serial equivalence** (parallel=false), the evaluation order MUST match the linked
list traversal order exactly.

---

## Error Handling

### force_stop Mechanism
- Before evaluation: if `force_stop && *force_stop`, skip the function call and assign `fmax`
- After evaluation: if `force_stop && *force_stop`, override `kret = -1` (setup error)
- The caller (DIRect.c) checks `force_stop` after `dirsamplef_` returns and sets `ierror = -102`

### kret Values (returned by dirinfcn_ / user function)
| kret | Meaning | f[2*pos+2] value | f[2*pos+1] value |
|------|---------|-----------------|-----------------|
| 0 | Feasible, evaluation succeeded | 0.0 | User function result |
| ≥ 1 | Infeasible (hidden constraint violated) | 2.0 | fmax (current max feasible) |
| -1 | Setup failure (or force_stop) | -1.0 | Unchanged (from dirinfcn_ or fmax if force_stop) |

### Feasibility Flag Semantics (f[2*pos+2])
- `0.0` = feasible, valid function value
- `2.0` = infeasible, f-value replaced with fmax
- `-1.0` = setup error, should not be used in optimization

---

## Comparison with DIRparallel.c

`DIRparallel.c` contains the same `direct_dirsamplef_()` function signature but uses PVM
(Parallel Virtual Machine) message passing:

1. **Master distributes**: sends points to worker processes via `mastersendif_()`
2. **Workers evaluate**: each worker evaluates one point and sends result back via `slavesendif_()`
3. **Master collects**: receives results via `masterrecvif_()`
4. **Master also evaluates**: when `datarec / nprocs` is an integer, master evaluates a point itself

Key differences from serial version:
- Uses `f[pos + f_dim1]` (column-major) instead of `f[(pos << 1) + 1]` (row-major transposed)
  — this is because the f2c-cleaned serial version uses SGJ's transposed layout while the
  parallel version uses the original Fortran column-major layout
- More complex evaluation order (interleaved master + slave evaluations)
- Same two-pass structure (evaluate, then update minf/minpos)

**The parallel version is NOT used in NLOPT** (requires PVM). Our Rust rayon parallelization
replaces both versions with a more modern approach.

---

## Memory Layout Details

### Center Coordinates Access Pattern
```c
x[i] = c__[i + pos * c_dim1]    // where c_dim1 = n
```
In 0-based terms: `c[pos * n + (i-1)]` — centers are stored contiguously per rectangle,
with dimension as the fast index. This is cache-friendly for copying one rectangle's coordinates.

### Function Value Access Pattern
```c
f[(pos << 1) + 1]    // f-value: f[2*pos + 1] (0-based: f[2*pos])
f[(pos << 1) + 2]    // flag:    f[2*pos + 2] (0-based: f[2*pos + 1])
```
The `<< 1` is equivalent to `* 2`. So function values and flags are interleaved:
`[f0_val, f0_flag, f1_val, f1_flag, f2_val, f2_flag, ...]`

### Linked List Traversal
```c
pos = point[pos]     // Follow linked list to next rectangle
```
The `point` array serves double duty:
- For allocated rectangles: next pointer in the depth-level sorted list
- For free rectangles: next pointer in the free list

---

## Rust Port Strategy

### Serial Path (parallel=false)
Implement as direct translation:
```rust
fn evaluate_sample_points(&mut self, new_pos: usize, maxi: usize) {
    let mut pos = new_pos;
    // Pass 1: evaluate all 2*maxi points
    for _ in 0..2*maxi {
        let x = self.storage.get_center(pos).to_vec();
        let (f_val, kret) = self.evaluate(&x);
        // Store results in f_values[pos]
        // Track fmax, ifeasiblef, iinfesiblef
        pos = self.storage.point[pos] as usize;
    }
    // Pass 2: update minf/minpos
    pos = new_pos;
    for _ in 0..2*maxi {
        if self.storage.f_values[pos*2] < self.minf && self.storage.f_values[pos*2+1] == 0.0 {
            self.minf = self.storage.f_values[pos*2];
            self.minpos = pos;
        }
        pos = self.storage.point[pos] as usize;
    }
}
```

### Parallel Path (parallel=true)
```rust
fn evaluate_sample_points_parallel(&mut self, new_pos: usize, maxi: usize) {
    // Collect all positions and their center coordinates
    let mut positions = Vec::with_capacity(2 * maxi);
    let mut pos = new_pos;
    for _ in 0..2*maxi {
        positions.push((pos, self.storage.get_center(pos).to_vec()));
        pos = self.storage.point[pos] as usize;
    }
    // Evaluate all points in parallel using rayon
    let results: Vec<(f64, i32)> = positions.par_iter()
        .map(|(_, x)| self.evaluate(x))
        .collect();
    // Process results sequentially (store f-values, update tracking vars)
    for (i, (pos, _)) in positions.iter().enumerate() {
        let (f_val, kret) = results[i];
        // Store and track as in serial version
    }
}
```

### Key Considerations
1. Each evaluation needs its own `x` buffer (already handled by copying centers)
2. `dirinfcn_()` modifies `x` in-place — Rust version should use a local copy
3. `fmax` used for infeasible replacement should be the pre-batch value in parallel mode
4. The `force_stop` check should be atomic in parallel mode
5. Pass 2 (minf update) is trivially parallelizable but probably not worth it (just 2*maxi comparisons)
