# NLOPT SGJ cdirect.c Analysis: Red-Black Tree DIRECT Re-implementation

## Source Files
- `nlopt/src/algs/cdirect/cdirect.c` (603 lines)
- `nlopt/src/algs/cdirect/cdirect.h` (80 lines)
- `nlopt/src/algs/cdirect/hybrid.c` (345 lines)
- `nlopt/src/algs/cdirect/README`

This is Steven G. Johnson's from-scratch re-implementation of the DIRECT and DIRECT-L
algorithms using red-black trees instead of the Gablonsky Fortran translation's linked-list
SoA approach.

---

## 1. `params` Struct and `which_alg` Encoding

### Definition (cdirect.c lines 54–83)

```c
typedef struct {
    int n;             // dimension
    int L;             // size of each rectangle = 2*n + 3
    double magic_eps;  // Jones' epsilon parameter (1e-4 recommended)
    int which_diam;    // 0 = Jones (Euclidean), 1 = Gablonsky (max-side)
    int which_div;     // 0 = Jones (all longest), 1 = Gablonsky, 2 = random longest
    int which_opt;     // 0 = Jones (all hull pts), 1 = DIRECT-L (one pt), 2 = randomized
    const double *lb, *ub;
    nlopt_stopping *stop;
    nlopt_func f; void *f_data;
    double *work;      // workspace >= 2*n doubles
    int *iwork;        // workspace >= n ints
    double minf, *xmin;
    rb_tree rtree;     // red-black tree sorted by (d, f, age)
    int age;           // age counter for next new rect
    double **hull;     // array for convex hull pointers
    int hull_len;      // allocated length of hull array
} params;
```

### `which_alg` Decoding (cdirect_unscaled, lines 489–491)

The single integer `which_alg` encodes three independent choices in base-3:

```c
p.which_diam = which_alg % 3;
p.which_div  = (which_alg / 3) % 3;
p.which_opt  = (which_alg / 9) % 3;
```

### `which_alg` Values for All NLOPT Algorithms

From `nlopt/src/api/optimize.c` lines 574–583, the `which_alg` is computed as:

```c
which_alg = (algorithm != NLOPT_GN_DIRECT)
          + 3 * (algorithm == NLOPT_GN_DIRECT_L_RAND ? 2 : (algorithm != NLOPT_GN_DIRECT))
          + 9 * (algorithm == NLOPT_GN_DIRECT_L_RAND ? 1 : (algorithm != NLOPT_GN_DIRECT));
```

Expanded for each algorithm:

| NLOPT Algorithm               | which_alg | which_diam | which_div | which_opt | Notes                          |
|-------------------------------|-----------|------------|-----------|-----------|--------------------------------|
| `NLOPT_GN_DIRECT`             | 0         | 0 (Jones)  | 0 (Jones) | 0 (all)   | Original DIRECT (scaled)       |
| `NLOPT_GN_DIRECT_L`           | 13        | 1 (Gabl)   | 1 (Gabl)  | 1 (one)   | DIRECT-L (scaled)              |
| `NLOPT_GN_DIRECT_L_RAND`      | 16        | 1 (Gabl)   | 2 (rand)  | 1 (one)   | DIRECT-L randomized (scaled)   |
| `NLOPT_GN_DIRECT_NOSCAL`      | 0         | 0 (Jones)  | 0 (Jones) | 0 (all)   | Original DIRECT (unscaled)     |
| `NLOPT_GN_DIRECT_L_NOSCAL`    | 13        | 1 (Gabl)   | 1 (Gabl)  | 1 (one)   | DIRECT-L (unscaled)            |
| `NLOPT_GN_DIRECT_L_RAND_NOSCAL`| 16       | 1 (Gabl)   | 2 (rand)  | 1 (one)   | DIRECT-L randomized (unscaled) |

**Note:** The `_NOSCAL` variants pass the same `which_alg` but call `cdirect_unscaled()` directly
instead of going through `cdirect()` which applies the rescaling wrapper.

### Verification of which_alg=13 for DIRECT-L

```
which_diam = 13 % 3 = 1     → Gablonsky (max-side)
which_div  = (13/3) % 3 = 1 → Gablonsky (cubes → all, rects → longest)
which_opt  = (13/9) % 3 = 1 → one point per diameter (skip duplicates)
```

### Verification of which_alg=16 for DIRECT-L-RAND

```
which_diam = 16 % 3 = 1     → Gablonsky (max-side)
which_div  = (16/3) % 3 = 2 → random longest side
which_opt  = (16/9) % 3 = 1 → one point per diameter
```

---

## 2. Hyperrectangle Memory Layout

Each hyperrectangle is a flat `double` array of length `L = 2n + 3`:

```
Index:  [0]       [1]      [2]     [3..n+2]      [n+3..2n+2]
Field:  diameter  f_value  age     centers[n]     widths[n]
```

- `[0]` = diameter (d): the "size" measure, rounded to float precision
- `[1]` = f: function value at center
- `[2]` = age: monotonically increasing counter for tie-breaking
- `[3..n+2]` = c[n]: center coordinates
- `[n+3..2n+2]` = w[n]: side widths (full width, NOT half-width)

Access patterns in code:
```c
double *c = rdiv + 3;       // center
double *w = c + n;           // widths (= rdiv + 3 + n)
```

---

## 3. `rect_diameter()` — Rectangle Measure (lines 94–112)

```c
static double rect_diameter(int n, const double *w, const params *p)
```

### Jones Measure (which_diam == 0)
Euclidean distance from center to vertex:
```
d = (float)(sqrt(sum(w[i]^2)) * 0.5)
```

### Gablonsky Measure (which_diam == 1)
Half of the longest side:
```
d = (float)(max(w[i]) * 0.5)
```

### Critical Detail: Float Rounding
Both measures cast the result to `float` before returning as `double`. This is a
**performance hack** for convex_hull(): since all rectangles in DIRECT fall into a
small number of diameter values (determined by trisection), casting to float ensures
that rects with the same "logical" diameter get the same numerical diameter despite
double-precision rounding differences. This enables the "vertical line" optimization
in convex_hull().

---

## 4. `sort_fv_compare()` and `sort_fv()` — Dimension Sorting (lines 116–134)

```c
static int sort_fv_compare(void *fv_, const void *a_, const void *b_)
static void sort_fv(int n, double *fv, int *isort)
```

Used in `divide_rect()` to sort dimensions by `min(f_plus, f_minus)` — the minimum
of the function values sampled at `center ± delta` along each dimension.

- `fv[2*i]` = f(center - delta_i)
- `fv[2*i+1]` = f(center + delta_i)
- Sort key: `min(fv[2*i], fv[2*i+1])`
- `isort[j]` = index of the j-th best dimension
- Uses `nlopt_qsort_r` (reentrant qsort with context pointer)

Dimensions NOT being divided get `fv[2*i] = fv[2*i+1] = HUGE_VAL`, so they sort last.

---

## 5. `function_eval()` — Objective Evaluation (lines 136–144)

```c
static double function_eval(const double *x, params *p)
```

Simple wrapper:
1. Call `p->f(p->n, x, NULL, p->f_data)` (gradient = NULL)
2. If `f < p->minf`, update `p->minf` and copy x to `p->xmin`
3. Increment `*(p->stop->nevals_p)`

### `FUNCTION_EVAL` Macro (line 145)

After evaluating, checks stopping conditions in order:
1. `nlopt_stop_forced()` → `NLOPT_FORCED_STOP`
2. `p->minf < p->stop->minf_max` → `NLOPT_MINF_MAX_REACHED`
3. `nlopt_stop_evals()` → `NLOPT_MAXEVAL_REACHED`
4. `nlopt_stop_time()` → `NLOPT_MAXTIME_REACHED`

On any stop, frees `freeonerr` (can be 0/NULL for no-op) and returns.

---

## 6. `divide_rect()` — Rectangle Trisection (lines 152–243)

```c
static nlopt_result divide_rect(double *rdiv, params *p)
```

This is the core division routine. It takes a rectangle `rdiv` and divides it
by trisection along one or more of its longest dimensions.

### Step 1: Find Longest Dimensions (lines 159–168)

```c
double wmax = w[0];
int imax = 0, nlongest = 0;
for (i = 1; i < n; ++i)
    if (w[i] > wmax) wmax = w[imax = i];
for (i = 0; i < n; ++i)
    if (wmax - w[i] <= wmax * EQUAL_SIDE_TOL)  // 5e-2 tolerance
        ++nlongest;
```

- `wmax` = longest side width
- `imax` = index of first longest side
- `nlongest` = count of sides within 5% of wmax
- `EQUAL_SIDE_TOL = 5e-2` — sides within 5% of longest are considered "equal"

### Step 2: Choose Division Strategy (line 169)

```c
if (p->which_div == 1 || (p->which_div == 0 && nlongest == n))
```

Two paths:

#### Path A: Multi-dimension trisection (which_div==1, or which_div==0 with all sides equal)

This is the **Gablonsky/DIRECT-L approach** (and Jones approach for cubes).

For each longest dimension:
1. Sample f at `center ± w[i]*THIRD` (lines 176–182)
2. Store in fv[2*i] and fv[2*i+1]
3. Non-longest dimensions get `fv = HUGE_VAL`

Then:
1. Sort dimensions by `min(f+, f-)` using `sort_fv()` (line 187)
2. Find the rect in the rb-tree (line 188)
3. For each longest dimension in sorted order (line 190):
   - Trisect: `w[isort[i]] *= THIRD` (line 192)
   - Recompute diameter: `rdiv[0] = rect_diameter(n, w, p)` (line 193)
   - Update age: `rdiv[2] = p->age++` (line 194)
   - Resort the parent node in the tree (line 195)
   - Create 2 child rectangles (lines 196–207):
     - Allocate `rnew`, copy from parent
     - Offset center: `rnew[3 + isort[i]] += w[isort[i]] * (2*k-1)` (±1)
     - Set f-value from pre-sampled values: `rnew[1] = fv[2*isort[i]+k]`
     - Set age: `rnew[2] = p->age++`
     - Insert into rb-tree

**Key insight:** The parent is progressively narrowed (width trisected) for each
dimension in sorted order. Each trisection updates the parent in-place and creates
two children. Children inherit the already-narrowed parent widths. This means
dimensions divided first get progressively thinner rectangles.

#### Path B: Single-dimension trisection (which_div==0 with nlongest<n, or which_div==2)

**Jones original approach** (non-cube rects) or **randomized**.

- `which_div == 2`: randomly pick among longest sides using `nlopt_iurand(nlongest)` (lines 212–219)
- `which_div == 0`: pick `imax` (first longest side) (line 222)

Then (lines 223–241):
1. Find rect in rb-tree
2. Trisect: `w[i] *= THIRD`
3. Recompute diameter, update age, resort parent
4. Create 2 children:
   - Offset center: `rnew[3 + i] += w[i] * (2*k-1)`
   - **Evaluate function** at child center (unlike Path A where evaluation happens before sorting)
   - Set age, insert into rb-tree

**Critical difference from Path A:**
- Path A pre-samples all dimensions, sorts, then divides. Children get pre-computed f-values.
- Path B divides one dimension and evaluates children after division.

---

## 7. `convex_hull()` — Lower Convex Hull (lines 261–378)

```c
static int convex_hull(rb_tree *t, double **hull, int allow_dups)
```

Computes the lower convex hull of points `(diameter, f_value)` from rectangles
stored in the rb-tree. The rb-tree is sorted lexicographically by `(d, f, age)`,
so an in-order traversal gives points sorted by diameter then f-value.

### Algorithm: Modified Monotone Chain [Andrew, 1979]

#### Step 1: Get min/max nodes (lines 270–276)
```c
n = nlopt_rb_tree_min(t);    // smallest diameter, smallest f at that diameter
nmax = nlopt_rb_tree_max(t); // largest diameter
xmin = n->k[0];              // min diameter
yminmin = n->k[1];           // f-value at min diameter
xmax = nmax->k[0];           // max diameter
```

#### Step 2: Handle duplicates at xmin (lines 278–284)
If `allow_dups` (which_opt == 0, Jones), include ALL points at `(xmin, yminmin)`.
Otherwise, include only one.

#### Step 3: Find nmax = first node with x==xmax (lines 289–301)
Uses a **performance hack**: instead of iterating backwards from tree max, uses
`nlopt_rb_tree_find_gt()` with `kshift[0] = xmax * (1 - 1e-13)` to jump directly
to the first node at xmax. This works because the float-rounding in `rect_diameter()`
ensures no two distinct diameter values differ by less than ~1e-7.

#### Step 4: Compute min slope (line 304)
```c
minslope = (ymaxmin - yminmin) / (xmax - xmin);
```
Any point above the line from `(xmin, yminmin)` to `(xmax, ymaxmin)` cannot be
on the convex hull. This is a fast pre-filter.

#### Step 5: Skip past xmin duplicates (lines 307–318)
Same performance hack to jump to first node with `x > xmin`.

#### Step 6: Main sweep (lines 320–367)

For each node between xmin and xmax:
1. **Slope pre-filter** (line 322): skip if above the min-to-max line
2. **Vertical line hack** (lines 327–343): if same x as previous hull point:
   - If higher f: skip entire vertical line (jump to next diameter)
   - If equal f and allow_dups: add to hull
3. **Left turn test** (lines 347–365): remove hull points until we make a left turn.
   - Uses cross product: `(t1-t2) × (k-t2) >= 0` means left turn (keep)
   - Handles equal points by looking back past duplicates
4. Add point to hull

#### Step 7: Handle duplicates at xmax (lines 369–376)
If allow_dups, include all points at `(xmax, ymaxmin)`.

### Return Value
Number of hull points, with pointers stored in `hull[]`.

---

## 8. `small()` — Width Tolerance Check (lines 382–390)

```c
static int small(double *w, params *p)
```

Returns 1 if ALL widths are below tolerance:
```
w[i] <= max(xtol_abs[i], (ub[i] - lb[i]) * xtol_rel)
```

Used to check if a divided rectangle has become too small (xtol convergence).

---

## 9. `divide_good_rects()` — Select and Divide PO Rectangles (lines 392–458)

```c
static nlopt_result divide_good_rects(params *p)
```

This is the per-iteration core: identify potentially optimal (PO) rectangles
via convex hull, then divide them.

### Step 1: Grow hull array if needed (lines 399–403)
```c
if (p->hull_len < p->rtree.N) {
    p->hull_len += p->rtree.N;
    p->hull = realloc(p->hull, sizeof(double*) * p->hull_len);
}
```

### Step 2: Compute convex hull (line 404)
```c
nhull = convex_hull(&p->rtree, hull = p->hull, p->which_opt != 1);
```
`allow_dups = (which_opt != 1)`: Jones (0) and randomized (2) allow duplicate
hull points; DIRECT-L (1) does not.

### Step 3: For each hull point, apply epsilon test (lines 406–436)

For each hull point `hull[i]`:

1. Find slopes to neighbors:
   - `K1` = slope to previous non-equal point
   - `K2` = slope to next non-equal point
   - `K = max(K1, K2)` — use the steeper slope

2. **Epsilon test** (line 419):
   ```c
   hull[i][1] - K * hull[i][0] <= p->minf - magic_eps * fabs(p->minf)
   ```
   OR `ip == nhull` (last hull point, always divide the largest rect)

3. If potentially optimal: call `divide_rect(hull[i], p)` (line 422)
   - Track `xtol_reached` for all divided rects

4. **DIRECT-L skip** (lines 432–433): if `which_opt == 1`, skip to next unequal
   diameter point (`i = ip - 1`). This means only ONE rectangle per diameter class
   is divided.

5. **Randomized skip** (lines 434–435): if `which_opt == 2`, randomly decide
   whether to continue with another equal point.

### Step 4: Fallback if nothing was divided (lines 437–456)

If no rectangle was divided:
1. If `magic_eps != 0`: try again with `magic_eps = 0` (goto divisions)
2. If `magic_eps == 0`: **fallback heuristic** — find the largest rectangle
   (by diameter) with the smallest f-value and divide it. This is an undocumented
   heuristic that SGJ notes "seems to work well."

### Return Value
- `NLOPT_XTOL_REACHED` if all divided rects are now too small
- `NLOPT_SUCCESS` otherwise
- Error codes from `divide_rect()` propagated

---

## 10. `cdirect_hyperrect_compare()` — Red-Black Tree Ordering (lines 463–472)

```c
int cdirect_hyperrect_compare(double *a, double *b)
```

Lexicographic comparison of `(d, f, age)`:
1. Compare `a[0]` vs `b[0]` (diameter)
2. Compare `a[1]` vs `b[1]` (f-value)
3. Compare `a[2]` vs `b[2]` (age)
4. Tie-breaker: pointer comparison `(int)(a - b)` (should never be needed)

This ordering means:
- In-order traversal visits rects from smallest to largest diameter
- Within same diameter, from smallest to largest f-value
- Within same (d,f), oldest first

---

## 11. `cdirect_unscaled()` — Main Loop (lines 476–549)

```c
nlopt_result cdirect_unscaled(int n, nlopt_func f, void *f_data,
                              const double *lb, const double *ub,
                              double *x, double *minf,
                              nlopt_stopping *stop,
                              double magic_eps, int which_alg)
```

### Initialization (lines 483–528)

1. Decode `which_alg` into `which_diam`, `which_div`, `which_opt`
2. Initialize params: n, L=2n+3, bounds, stop, f, minf=HUGE_VAL, age=0
3. Allocate workspace: `work` (2n doubles), `iwork` (n ints), `hull` (128 pointers)
4. Initialize rb-tree with `cdirect_hyperrect_compare`
5. Create initial rectangle:
   - Center: `c[i] = 0.5 * (lb[i] + ub[i])` (midpoint of domain)
   - Width: `w[i] = ub[i] - lb[i]` (full domain width)
   - Diameter: `rect_diameter(n, w, p)`
   - Evaluate center: `function_eval(c, p)`
   - Age: `p->age++` (starts at 0)
   - Insert into rb-tree
6. Immediately divide the initial rect: `divide_rect(rnew, &p)`

### Main Loop (lines 531–539)

```c
while (1) {
    double minf0 = p.minf;
    ret = divide_good_rects(&p);
    if (ret != NLOPT_SUCCESS) goto done;
    if (p.minf < minf0 && nlopt_stop_f(p.stop, p.minf, minf0)) {
        ret = NLOPT_FTOL_REACHED;
        goto done;
    }
}
```

Very simple loop:
1. Save current minf
2. Call `divide_good_rects()` — select PO rects, divide them
3. If non-success return (error or xtol_reached), stop
4. If minf improved AND ftol reached, stop with NLOPT_FTOL_REACHED
5. Repeat

### Cleanup (lines 541–548)

1. Destroy rb-tree and free all rectangle key arrays
2. Free hull, iwork, work
3. Set output `*minf = p.minf`

### Stopping Conditions (summary)

| Condition              | Where checked           | Return code              |
|------------------------|-------------------------|--------------------------|
| force_stop             | FUNCTION_EVAL macro     | NLOPT_FORCED_STOP        |
| minf_max reached       | FUNCTION_EVAL macro     | NLOPT_MINF_MAX_REACHED   |
| maxeval reached        | FUNCTION_EVAL macro     | NLOPT_MAXEVAL_REACHED    |
| maxtime reached        | FUNCTION_EVAL macro     | NLOPT_MAXTIME_REACHED    |
| ftol reached           | main loop               | NLOPT_FTOL_REACHED       |
| xtol reached           | divide_good_rects()     | NLOPT_XTOL_REACHED       |
| out of memory          | ALLOC_RECT, realloc     | NLOPT_OUT_OF_MEMORY      |

---

## 12. `cdirect()` — Rescaling Wrapper (lines 555–603)

```c
nlopt_result cdirect(int n, nlopt_func f, void *f_data,
                     const double *lb, const double *ub,
                     double *x, double *minf,
                     nlopt_stopping *stop,
                     double magic_eps, int which_alg)
```

Wraps `cdirect_unscaled()` to map the original domain to a unit hypercube.

### Scaling Logic

1. Allocate `d.x`: buffer of `n * (xtol_abs ? 4 : 3)` doubles
2. Map initial guess: `x[i] = (x[i] - lb[i]) / (ub[i] - lb[i])`
3. Set scaled bounds: `lb_scaled = 0, ub_scaled = 1` (for all dims)
4. Scale xtol_abs if present: `xtol_abs_scaled[i] = xtol_abs[i] / (ub[i] - lb[i])`
5. Temporarily replace `stop->xtol_abs` with scaled version
6. Call `cdirect_unscaled()` with scaled bounds and `cdirect_uf` wrapper
7. Restore `stop->xtol_abs`
8. Unscale result: `x[i] = lb[i] + x[i] * (ub[i] - lb[i])`

### `cdirect_uf()` — Unscaling Function Wrapper (lines 555–567)

```c
double cdirect_uf(unsigned n, const double *xu, double *grad, void *d_)
```

Maps unit-cube coordinates back to original domain for function evaluation:
```c
d->x[i] = d->lb[i] + xu[i] * (d->ub[i] - d->lb[i]);
```

If gradient is provided (never in DIRECT, but API supports it):
```c
grad[i] *= d->ub[i] - d->lb[i];
```

---

## 13. `cdirect_hyperrect_compare()` — Exported Compare Function (lines 463–472)

This is declared in cdirect.h and also used by hybrid.c. It provides the lexicographic
ordering `(diameter, f_value, age)` for the red-black tree, ensuring:
- Smallest-diameter rects are at the tree minimum
- Largest-diameter rects are at the tree maximum
- Within same diameter, lowest-f rects come first
- Age breaks remaining ties (oldest first)

---

## 14. Key Constants

| Constant        | Value                    | Purpose                             |
|-----------------|--------------------------|-------------------------------------|
| `THIRD`         | 0.333333...              | Trisection factor (1/3)             |
| `EQUAL_SIDE_TOL`| 5e-2                     | Tolerance for equating side lengths |
| `1e-13`         | float rounding threshold | Convex hull performance hack        |

---

## 15. Key Differences from Gablonsky Translation (DIRect.c)

| Aspect              | Gablonsky (DIRect.c)           | SGJ (cdirect.c)                  |
|---------------------|--------------------------------|----------------------------------|
| Data structure      | SoA + linked lists             | Red-black tree                   |
| Memory              | Pre-allocated arrays           | Dynamic malloc per rect          |
| Rectangle layout    | Separate arrays (c[], f[], length[]) | Single flat array [d,f,age,c,w] |
| Side length repr.   | Integer indices into thirds[]  | Actual widths (doubles)          |
| PO selection        | Custom sweep (dirchoose_)      | Convex hull (monotone chain)     |
| Diameter            | Computed from length indices   | Stored explicitly in rect[0]     |
| Equal-value handling| dirdoubleinsert_() for Jones   | allow_dups in convex_hull()      |
| Infeasible points   | dirreplaceinf_()               | Not handled (no NaN support)     |
| Epsilon update      | iepschange flag                | Retry with eps=0 fallback        |
| Scaling formula     | xs1=u-l, xs2=l/(u-l)          | x_scaled = (x-lb)/(ub-lb)       |
| Algorithm variants  | algmethod (0 or 1)             | which_alg (base-3 encoding)      |
| Randomized variant  | Not supported                  | which_div=2, which_opt=2         |

---

## 16. Algorithm Flow Summary

```
cdirect_unscaled():
  1. Parse which_alg → which_diam, which_div, which_opt
  2. Allocate workspace (work, iwork, hull)
  3. Init rb-tree with lexicographic compare
  4. Create initial rect at domain center
  5. Evaluate center, insert into tree
  6. divide_rect() on initial rect (first trisection)
  7. MAIN LOOP:
     a. Save minf0
     b. divide_good_rects():
        i.   Grow hull array if needed
        ii.  convex_hull() on rb-tree → hull points
        iii. For each hull point:
             - Compute slopes K1, K2 to neighbors
             - Epsilon test: f - K*d <= minf - eps*|minf|
             - If PO or last point: divide_rect()
             - DIRECT-L: skip equal-diameter points
        iv.  Fallback: if nothing divided:
             - eps=0 retry OR divide largest rect
     c. Check ftol convergence
     d. Repeat until stopping condition
  8. Cleanup: destroy tree, free memory
  9. Return best x and minf
```

---

## 17. Red-Black Tree Usage

The rb-tree stores rectangle pointers as keys, sorted by `(d, f, age)`:

- `nlopt_rb_tree_init()` — Initialize with comparator
- `nlopt_rb_tree_insert()` — Insert new rect
- `nlopt_rb_tree_find()` — Find exact rect pointer
- `nlopt_rb_tree_resort()` — Re-sort after modifying key (diameter changes during division)
- `nlopt_rb_tree_min()` — Smallest diameter rect
- `nlopt_rb_tree_max()` — Largest diameter rect
- `nlopt_rb_tree_succ()` — Next in-order (larger diameter or larger f at same diameter)
- `nlopt_rb_tree_pred()` — Previous in-order
- `nlopt_rb_tree_find_gt()` — First node greater than key (for performance hacks)
- `nlopt_rb_tree_destroy_with_keys()` — Free all nodes AND their key arrays

### Rust Equivalent

A `BTreeMap<RectKey, HyperRect>` where `RectKey` implements `Ord` for the
`(diameter, f_value, age)` lexicographic ordering provides equivalent functionality.
Alternatively, a `BTreeSet<HyperRect>` where `HyperRect` implements `Ord`.

---

## 18. Edge Cases and Subtle Behaviors

1. **Initial division**: The initial rectangle is divided immediately after creation
   (line 528), before the main loop starts. This means the first main-loop iteration
   sees the tree with at least 3 rects (parent + 2 children from initial trisection).

2. **Age counter**: Ages increase monotonically. They serve only as tie-breakers in
   the tree ordering. They ensure that newer rects with identical (d,f) go after
   older ones.

3. **Division modifies parent in-place**: `divide_rect()` modifies the parent's width,
   diameter, and age, then calls `nlopt_rb_tree_resort()` to fix tree ordering.
   Children are allocated fresh.

4. **Float rounding in diameter**: Critical for convex hull performance. Without it,
   rects at the same "logical" diameter level could get slightly different double
   values, destroying the vertical-line optimization.

5. **Fallback heuristic**: When no rect passes the epsilon test, the algorithm tries
   eps=0, and if still nothing, divides the largest rect with smallest f. This
   undocumented heuristic prevents the algorithm from stalling.

6. **No infeasible point handling**: Unlike the Gablonsky translation which has
   `dirreplaceinf_()`, cdirect.c has no mechanism for handling NaN/Inf function
   values. Functions must be defined everywhere in the domain.

7. **Function evaluation during division**: In Path A (multi-dim), function values
   are computed BEFORE sorting and division. In Path B (single-dim), function
   values are computed AFTER division (inside the child creation loop).
