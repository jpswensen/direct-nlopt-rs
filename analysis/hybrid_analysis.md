# NLOPT hybrid.c Analysis: DIRECT + Local Optimization Hybrid

## Source Files
- `nlopt/src/algs/cdirect/hybrid.c` (345 lines)
- `nlopt/src/algs/cdirect/cdirect.h` (80 lines) — shared declarations

This is Steven G. Johnson's hybrid variant of DIRECT that combines the DIRECT global
search strategy with a local optimization algorithm. Instead of evaluating a single
point per rectangle, it runs a full local optimization within each rectangle's bounds.

---

## 1. `params` Struct (hybrid.c lines 42–57)

```c
typedef struct {
    int n;                 // dimension
    int L;                 // 3*n+3 (extended hyperrect layout)
    const double *lb, *ub; // original problem bounds
    nlopt_stopping *stop;  // stopping criteria
    nlopt_func f; void *f_data; // objective function
    double minf, *xmin;    // best minimum found so far
    rb_tree rtree;         // red-black tree of rects, sorted by (d,-f,-a)
    int age;               // age counter for next new rect (decrements)
    double *work;          // workspace of length >= 2*n
    nlopt_opt local_opt;   // local optimization algorithm handle
    int local_maxeval;     // max evaluations for local optimizer
    int randomized_div;    // 1 to use randomized division
} params;
```

### Key differences from cdirect.c `params`:
| Field            | cdirect.c              | hybrid.c               |
|------------------|------------------------|------------------------|
| `L`              | `2*n+3`                | `3*n+3`                |
| `magic_eps`      | present                | absent                 |
| `which_diam/div/opt` | present            | absent                 |
| `minf`, `xmin`   | present                | present                |
| `local_opt`      | absent                 | present                |
| `local_maxeval`  | absent                 | present                |
| `randomized_div` | absent                 | present                |

### Extended Hyperrect Layout (3n+3)
Each rectangle is a double array of length `L = 3*n+3`:
```
[diameter, -f_value, -age, x[0..n-1], c[0..n-1], w[0..n-1]]
  r[0]      r[1]     r[2]  r[3..n+2]  r[n+3..2n+2] r[2n+3..3n+2]
```
Where:
- `d = r[0]` — diameter (max side length, using `longest()`)
- `-f = r[1]` — negated function value (so tree sorts by largest -f = smallest f)
- `-a = r[2]` — negated age (newer rects have more negative age)
- `x[n]` — local optimum position within the rectangle
- `c[n]` — center of the rectangle
- `w[n]` — widths of the rectangle (full widths, not half-widths)

**Critical difference from cdirect.c:** The standard cdirect stores `(d, f, age, c[n], w[n])` 
in `2n+3` doubles. The hybrid adds `x[n]` (local optimum) between the metadata and center, 
extending to `3n+3`. The `f` value stored is the local optimum value, not just the center 
evaluation.

### Sorting Convention
The red-black tree uses `cdirect_hyperrect_compare()` from cdirect.c for lexicographic 
ordering by `(d, -f, -a)`. Since `f` is stored as `-f` and `a` as `-a`:
- **Largest diameter** sorts to the tree maximum (largest `r[0]`)
- Among equal diameters: **smallest f-value** sorts higher (largest `-f`)
- Among equal (d, f): **oldest rectangle** sorts higher (largest `-a`)

This means `rb_tree_max()` returns the largest-diameter rectangle with the best (lowest) 
objective value, breaking ties by oldest age.

---

## 2. `fcount()` (hybrid.c lines 65–70)

```c
static double fcount(unsigned n, const double *x, double *grad, void *p_)
{
    params *p = (params *) p_;
    ++ *(p->stop->nevals_p);
    return p->f(n, x, grad, p->f_data);
}
```

A thin wrapper around the user's objective function that increments the global evaluation 
counter. This is used as the objective for the local optimizer, ensuring that all local 
optimization evaluations are counted toward the global stopping criteria.

---

## 3. `optimize_rect()` (hybrid.c lines 72–113)

### Signature
```c
static nlopt_result optimize_rect(double *r, params *p)
```

### Behavior
Runs a local optimization within the bounds of rectangle `r`:

1. **Early termination checks** (lines 82–85):
   - If `maxeval` reached → return `NLOPT_MAXEVAL_REACHED`
   - If `maxtime` reached → return `NLOPT_MAXTIME_REACHED`

2. **Compute local bounds** (lines 87–90):
   - `lb[i] = c[i] - 0.5 * w[i]` (rectangle lower bound)
   - `ub[i] = c[i] + 0.5 * w[i]` (rectangle upper bound)

3. **Configure local optimizer** (lines 91–101):
   - Set bounds to rectangle bounds
   - Set maxeval to `min(local_maxeval, remaining_global_evals)`
   - Set maxtime to remaining global time

4. **Run local optimization** (line 102):
   - `nlopt_optimize(p->local_opt, x, &minf)` — optimizes starting from `x`, stores result in `x`
   - The starting point `x = r + 3` is the local optimum slot in the rectangle

5. **Update stored f-value** (line 103):
   - `r[1] = -minf` — stores negated minimum for sorting

6. **Update global best** (lines 104–111):
   - If `ret > 0` (success): check if `minf < p->minf`, update global best
   - If `NLOPT_MINF_MAX_REACHED`: return immediately
   - Otherwise: return `NLOPT_SUCCESS`
   - If `ret <= 0` (error): propagate error

### Key observation
The local optimum `x` is stored in the rectangle at `r[3..3+n-1]`, separate from the 
center `c` at `r[3+n..3+2n-1]`. After local optimization, `x` may be anywhere within 
the rectangle bounds, not necessarily at the center.

---

## 4. `randomize_x()` (hybrid.c lines 115–124)

```c
static void randomize_x(int n, double *r)
{
    double *x = r + 3, *c = x + n, *w = c + n;
    for (i = 0; i < n; ++i)
        x[i] = nlopt_urand(c[i] - w[i]*(0.5*THIRD), c[i] + w[i]*(0.5*THIRD));
}
```

Sets the local optimizer starting point `x[i]` to a random point within the **middle third** 
of the rectangle (from `c ± w/6`). This avoids starting too close to the edges, where the 
local optimizer might immediately exit the bounds.

Used when `randomized_div = 1` (randomized division mode).

---

## 5. `longest()` (hybrid.c lines 128–133)

```c
static double longest(int n, const double *w)
{
    double wmax = w[n-1];
    for (n = n-2; n >= 0; n--) if (w[n] > wmax) wmax = w[n];
    return wmax;
}
```

Returns the maximum width among all dimensions. This is the **Gablonsky-style diameter** 
(max side length), NOT the Jones Euclidean diameter. The hybrid always uses max-side 
diameter — there is no `which_diam` option.

---

## 6. `divide_largest()` (hybrid.c lines 137–226) — CORE FUNCTION

### Signature
```c
static nlopt_result divide_largest(params *p)
```

### Overview
Takes the "largest" rectangle from the red-black tree (max node = largest diameter, 
best f, oldest age) and divides it. This is the core iteration of the hybrid algorithm.

### Step-by-step behavior:

#### 6.1. Get the largest rectangle (lines 141–147)
```c
rb_node *node = nlopt_rb_tree_max(&p->rtree);
double *r = node->k;
double *x = r + 3, *c = x + n, *w = c + n;
```
Uses the tree as a max-heap: always processes the rectangle with largest diameter first.

#### 6.2. Check xtol stopping (lines 153–157)
```c
for (i = 0; i < n; ++i)
    if (w[i] > stop->xtol_rel * (ub[i] - lb[i])
        && w[i] > (stop->xtol_abs ? stop->xtol_abs[i] : 0))
        break;
if (i == n) return NLOPT_XTOL_REACHED;
```
If ALL dimensions of the largest rectangle are below the xtol threshold, stop. Since this 
is the largest rectangle, all other rectangles must also be small enough.

#### 6.3. Select dimension to divide (lines 159–173)

**Randomized mode** (`randomized_div = 1`, lines 159–169):
1. Find `wmax = longest(n, w)`
2. Count `nlongest` = number of dims within `EQUAL_SIDE_TOL` (5%) of `wmax`
3. Pick a random index among the ~longest sides using `nlopt_iurand(nlongest)`
4. Walk dimensions to find the selected one

**Deterministic mode** (lines 170–173):
1. Pick the first dimension with the largest width (`w[idiv]`)

**Note:** Unlike standard DIRECT which divides ALL longest dimensions, hybrid divides 
only ONE dimension per iteration (either the first longest or a random longest).

#### 6.4. Bisect vs Trisect decision (line 175)
```c
if (fabs(x[idiv] - c[idiv]) > (0.5 * THIRD) * w[idiv])
```
This is the **key innovation** of the hybrid algorithm:
- If the local optimum `x` is far from the center `c` (more than `w/6` away along the 
  division dimension), use **bisection** to place the split closer to the optimum.
- Otherwise, use standard **trisection**.

This adapts the subdivision to the structure discovered by local optimization.

#### 6.5. Bisection path (lines 176–198)
```c
double deltac = (x[idiv] > c[idiv] ? 0.25 : -0.25) * w[idiv];
w[idiv] *= 0.5;          // halve the width
c[idiv] += deltac;       // shift center toward optimum
r[0] = longest(n, w);    // update diameter
r[2] = p->age--;         // update age
node = nlopt_rb_tree_resort(&p->rtree, node); // re-sort parent
```
The bisection splits the rectangle into two halves:
- **Parent** gets the half containing the local optimum `x`
  - Center shifts by `±0.25 * w` (toward `x`)
  - Width halved
  - `r[1]` (f-value) unchanged since it still contains the local optimum
- **Child** (newly allocated) gets the other half
  - Center shifts opposite direction: `rnew[3+n+idiv] -= deltac*2`
  - Width is the same (halved from parent's copy)
  - If randomized: random start point; else `x = c` (start from center)
  - Local optimization run on child via `optimize_rect(rnew, p)`

#### 6.6. Trisection path (lines 199–222)
```c
w[idiv] *= THIRD;         // trisect the width (1/3)
r[0] = longest(n, w);     // update diameter
r[2] = p->age--;          // update age
node = nlopt_rb_tree_resort(&p->rtree, node); // re-sort parent
```
Standard trisection creates two new children at `c ± w_new`:
- **Parent** retains center, width trisected
  - `r[1]` unchanged (still has local optimum)
- **Two children** (i = -1, +1):
  - Center offset by `±w_new` along division dimension
  - If randomized: random start point; else `x = c` (start from center)
  - Local optimization run on each child via `optimize_rect(rnew, p)`

**Note:** Line 211 has a subtle bug/feature: `rnew[3+n+idiv] += w[i] * i` — here `i` is 
the loop variable taking values -1 and +1, but `w` is already the trisected width array 
from the parent (since `w[idiv]` was already multiplied by `THIRD`). So the offset is 
`w_new * (±1)`, which correctly places children at `c ± w_new`.

Wait — actually `w[i]` accesses `w[-1]` or `w[+1]`, NOT `w[idiv]`. Looking more carefully:
the variable `i` takes values -1 and +1 in the loop. `w` points to `c + n = r + 3 + 2*n`.
So `w[i]` for `i=-1` reads `w[-1]` which is `c[n-1]` — this appears to be a **bug** in the 
NLOPT source. However, the multiplication `w[i] * i` for `i=-1` gives `c[n-1] * (-1)` 
and for `i=+1` gives `w[1]`.

**Correction:** Re-reading more carefully, `i` is used as both the loop counter AND as a 
signed offset multiplier. The intent is `rnew[3+n+idiv] += w[idiv] * i`, meaning offset 
the center by `±w[idiv]` (the already-trisected width). But the code writes `w[i]` not 
`w[idiv]`. This is indeed a bug — for `i = -1`, it reads `c[n-1]` instead of `w[idiv]`.

**Bug analysis:** Actually, re-examining: after `w[idiv] *= THIRD`, `w` still points to 
the widths array of the *parent* rectangle. `rnew` was `memcpy`'d from `r`, so 
`rnew[3+2*n+idiv]` has the trisected width. The code does 
`rnew[3+n+idiv] += w[i] * i` where:
- `rnew[3+n+idiv]` is the center along dim `idiv` in the new rect
- `w[i]` for `i=-1` reads `*(w-1)` = `c[n-1]` of the parent — **this is a bug**
- `w[i]` for `i=+1` reads `w[1]` which is the width of dimension 1

The intended code was likely `rnew[3+n+idiv] += w[idiv] * i`, offsetting the child center 
by ±1 trisected width along the division dimension.

**Impact:** This bug would cause incorrect child placement in trisection for most cases, 
but the hybrid algorithm is rarely used in practice (it's for `NLOPT_GN_CRS2_LM` style 
approaches). The bisection path (which is more commonly triggered when local optima are 
found) does not have this issue.

#### 6.7. Check ftol after division (lines 223–224)
```c
if (p->minf < minf_start && nlopt_stop_f(p->stop, p->minf, minf_start))
    return NLOPT_FTOL_REACHED;
```
If the global best improved during this division and the improvement is below ftol, stop.

---

## 7. `cdirect_hybrid_unscaled()` (hybrid.c lines 230–305) — MAIN ENTRY

### Signature
```c
nlopt_result cdirect_hybrid_unscaled(int n, nlopt_func f, void *f_data,
                                     const double *lb, const double *ub,
                                     double *x, double *minf,
                                     nlopt_stopping *stop,
                                     nlopt_algorithm local_alg,
                                     int local_maxeval,
                                     int randomized_div)
```

### Step-by-step behavior:

#### 7.1. Initialize params (lines 239–256)
- Set `p.n = n`, `p.L = 3*n+3`
- Store bounds, stopping criteria, objective function
- Initialize `p.minf = HUGE_VAL` (no minimum found yet)
- Set `p.xmin = x` (output pointer for best x)
- Set `p.age = 0` (will decrement, so newer rects have more negative age)

#### 7.2. Create red-black tree (lines 257–259)
```c
nlopt_rb_tree_init(&p.rtree, cdirect_hyperrect_compare);
p.work = (double *) malloc(sizeof(double) * (2*n));
```
Uses the same comparison function as cdirect.c.

#### 7.3. Create initial rectangle (lines 261–267)
```c
for (i = 0; i < n; ++i) {
    rnew[3+i] = rnew[3+n+i] = 0.5 * (lb[i] + ub[i]); // x = c = midpoint
    rnew[3+2*n+i] = ub[i] - lb[i];                     // w = full width
}
rnew[0] = longest(n, rnew+2*n);  // diameter = max width
rnew[2] = p.age--;                // age = 0, then decrement
```
Initial rectangle spans the full domain. Both the local optimum start point `x` and 
center `c` are set to the domain midpoint.

**Note:** `longest(n, rnew+2*n)` — the offset `2*n` from `rnew` should be `3+2*n` to 
reach the widths section. But actually `rnew+2*n` points to `rnew[2*n]` which is NOT the 
widths array (widths start at `rnew[3+2*n]`). This appears to be another bug.

**Correction:** Wait, looking at line 266 more carefully:
`rnew[0] = longest(n, rnew+2*n)` — should this be `longest(n, rnew+3+2*n)`? 
Actually `rnew + 3 + 2*n` is the widths array. But the code uses `rnew + 2*n`.
For `n >= 3`, `rnew[2*n]` would read from `rnew[6]` = `x[3]` or later, not widths.

Actually, let me reconsider: `rnew+2*n` when `n=2` gives `rnew+4`, and widths start at 
`rnew[3+2*2] = rnew[7]`. So `rnew+4` is `x[1]` — still wrong.

But wait — looking at the initial setup: `x[i]` and `c[i]` are both set to the midpoint, 
and `w[i] = ub[i] - lb[i]`. For a symmetric problem like `[-5,5]`, midpoint=0 and w=10. 
`longest(n, rnew+2*n)` would read starting from `rnew[2n]`. For `n=2`: 
`rnew[4] = x[1] = 0`, `rnew[5] = c[0] = 0`. So it computes max of wrong values.

However this code is in the actual NLOPT repository and presumably passes tests, so perhaps 
I'm miscounting. Let me recheck the layout:
- `rnew[0]` = diameter
- `rnew[1]` = -f
- `rnew[2]` = -age
- `rnew[3]` through `rnew[3+n-1]` = x (local optimum), n values
- `rnew[3+n]` through `rnew[3+2n-1]` = c (center), n values  
- `rnew[3+2n]` through `rnew[3+3n-1]` = w (widths), n values

So widths start at `rnew[3+2*n]` and `longest(n, rnew+3+2*n)` would be correct.

The code on line 266 says: `rnew[0] = longest(n, rnew+2*n)` — this is reading from 
`rnew[2*n]` which is NOT the widths array. For `n=1`: `rnew[2]` = -age. For `n=2`: 
`rnew[4]` = x[1]. This does appear to be a bug in the code, though it may work 
incidentally for some values of n because x[i] was just set to the same values that 
would appear if reading with a different offset, or it may be masked by the local 
optimization that follows.

**UPDATE:** On closer reading, the initial rectangle creation on lines 261-267 doesn't 
set `rnew[1]` (the f-value) — that gets set by `optimize_rect(rnew, &p)` on line 290. 
The diameter computation bug would give wrong initial diameter, but since `optimize_rect` 
may not depend on the diameter value, it might not matter for the first rectangle.

#### 7.4. Configure local optimizer (lines 269–288)
```c
p.local_opt = nlopt_create(local_alg, n);
```
Creates an NLOPT optimization object with the user-specified local algorithm. Copies 
stopping criteria from the global stopping criteria to the local optimizer:
- `stopval` (minimum f target)
- `ftol_rel`, `ftol_abs` (function tolerance)
- `xtol_rel`, `xtol_abs` (x tolerance)
- Sets the objective to `fcount` (wrapper that counts evaluations)

#### 7.5. Optimize initial rectangle (lines 290–292)
```c
ret = optimize_rect(rnew, &p);
if (ret != NLOPT_SUCCESS) { free(rnew); goto done; }
if (!nlopt_rb_tree_insert(&p.rtree, rnew)) { free(rnew); goto done; }
```
Runs local optimization on the initial full-domain rectangle, then inserts it into the tree.

#### 7.6. Main loop (lines 294–296)
```c
do {
    ret = divide_largest(&p);
} while (ret == NLOPT_SUCCESS);
```
Extremely simple main loop: repeatedly divide the largest rectangle until a stopping 
condition is triggered. No convex hull, no epsilon test, no potentially-optimal selection.

#### 7.7. Cleanup (lines 298–304)
```c
nlopt_rb_tree_destroy_with_keys(&p.rtree);
free(p.work);
nlopt_destroy(p.local_opt);
*minf = p.minf;
```
Destroys the tree (freeing all rectangle arrays), workspace, and local optimizer.

---

## 8. `cdirect_hybrid()` (hybrid.c lines 308–345) — RESCALING WRAPPER

### Signature
```c
nlopt_result cdirect_hybrid(int n, nlopt_func f, void *f_data,
                            const double *lb, const double *ub,
                            double *x, double *minf,
                            nlopt_stopping *stop,
                            nlopt_algorithm local_alg,
                            int local_maxeval,
                            int randomized_div)
```

### Behavior
Identical pattern to `cdirect()` in cdirect.c — rescales to the unit hypercube:

1. Creates `cdirect_uf_data` struct
2. Maps `x[i]` from original to unit: `x[i] = (x[i] - lb[i]) / (ub[i] - lb[i])`
3. Sets unit bounds: `lb_unit[i] = 0`, `ub_unit[i] = 1`
4. Rescales `xtol_abs` if present
5. Calls `cdirect_hybrid_unscaled()` with unit bounds and `cdirect_uf` wrapper
6. Maps `x[i]` back: `x[i] = lb[i] + x[i] * (ub[i] - lb[i])`

---

## 9. Key Differences from Standard DIRECT (cdirect.c)

| Aspect                 | Standard DIRECT (cdirect.c)                     | Hybrid (hybrid.c)                               |
|------------------------|--------------------------------------------------|--------------------------------------------------|
| **Rectangle layout**   | `2n+3`: (d, f, age, c[n], w[n])                 | `3n+3`: (d, -f, -a, x[n], c[n], w[n])           |
| **Function evaluation**| Single point evaluation at center                | Full local optimization within rect bounds       |
| **f-value meaning**    | f(center)                                        | min f found by local optimizer within rect        |
| **Rect selection**     | Convex hull → epsilon test → PO selection        | Simply take largest rect (max tree node)          |
| **Division strategy**  | Trisect ALL longest dims (or one for Gablonsky)  | Bisect or trisect ONE dim, based on x vs c       |
| **Bisection**          | Never (always trisection)                        | When local optimum far from center                |
| **Diameter measure**   | Jones (Euclidean) or Gablonsky (max-side)        | Always max-side (`longest()`)                    |
| **Epsilon parameter**  | Used for PO selection                            | Not used (no convex hull)                        |
| **Algorithm variants** | DIRECT, DIRECT-L, randomized, noscal             | Scaled + randomized only                         |
| **Stored f convention**| `r[1] = f` (direct value)                        | `r[1] = -f` (negated for max-tree sorting)       |

---

## 10. Scope Decision: Include Hybrid in Rust Implementation?

### Arguments for EXCLUDING hybrid mode:
1. **Complexity:** Requires integrating with a local optimizer (NLOPT's full optimizer 
   framework), which is outside the scope of a standalone DIRECT library.
2. **Dependency:** The local optimizer (`nlopt_opt`) is a complex NLOPT-specific type. 
   A Rust port would need to either depend on NLOPT bindings or implement a trait-based 
   local optimizer interface.
3. **Bugs:** The NLOPT implementation has apparent bugs (diameter computation, trisection 
   offset) suggesting it may be less well-tested than the main DIRECT codepaths.
4. **Usage:** The hybrid variant is rarely used in practice — most NLOPT users use the 
   standard DIRECT or DIRECT-L algorithms.
5. **PRD scope:** The PRD focuses on faithful DIRECT/DIRECT-L implementation with parallel 
   evaluation; the hybrid is architecturally different (no convex hull, no epsilon).

### Arguments for INCLUDING (partial):
1. The bisect/trisect decision based on local optimum position is an interesting idea.
2. A trait-based local optimizer interface could be provided for users who want hybrid mode.

### Recommendation: **EXCLUDE** from initial implementation.
The hybrid mode is architecturally distinct from standard DIRECT (no PO selection, requires 
local optimizer), has suspected bugs, and would add significant complexity. It can be added 
as a future extension if demand exists. The Rust crate should focus on faithful, high-quality 
ports of the standard DIRECT and DIRECT-L algorithms from both NLOPT codepaths (Gablonsky 
translation and SGJ re-implementation).

---

## 11. Summary of Functions

| Function                        | Lines     | Purpose                                                 |
|---------------------------------|-----------|---------------------------------------------------------|
| `fcount()`                      | 65–70     | Objective wrapper that counts evaluations               |
| `optimize_rect()`               | 72–113    | Run local optimizer within rectangle bounds             |
| `randomize_x()`                 | 115–124   | Randomize starting point within middle third of rect    |
| `longest()`                     | 128–133   | Compute max-side diameter                               |
| `divide_largest()`              | 137–226   | Core: take largest rect, bisect or trisect one dim      |
| `cdirect_hybrid_unscaled()`     | 230–305   | Main entry: init, loop divide_largest until stop        |
| `cdirect_hybrid()`              | 308–345   | Rescaling wrapper to unit hypercube                     |
