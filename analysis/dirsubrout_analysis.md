# NLOPT Gablonsky Subroutines Analysis

## Source File: `nlopt/src/algs/direct/DIRsubrout.c`

This file is a Fortran-to-C translation (via f2c, hand-cleaned by SGJ) of Gablonsky's
DIRECT subroutines. It contains all helper functions called by the main loop in
`direct_direct_()` (DIRect.c). All functions use **1-based Fortran indexing** internally
via pointer adjustment macros (`--x;`, `++anchor;`, etc.).

---

## 1. `direct_dirgetlevel_()` — Rectangle Level Computation

### Signature
```c
integer direct_dirgetlevel_(integer *pos, integer *length, integer *maxfunc,
                            integer *n, integer jones)
```

### Parameters
| Parameter | Direction | Description |
|-----------|-----------|-------------|
| `pos` | in | Position (1-based index) of the rectangle in the length array |
| `length` | in | `[n × MAXFUNC]` array of side-length indices |
| `maxfunc` | in | Leading dimension of length (unused after adjustment) |
| `n` | in | Problem dimension |
| `jones` | in | 0 = Gablonsky (DIRECT-L), 1 = Jones Original (DIRECT) |

### Behavior

**When `jones == 0` (Gablonsky / DIRECT-L):**
The level encodes both the minimum side-length index `k` and how many dimensions
share that minimum. The formula is:

1. Let `help` = `length[pos, 1]` (side-length index of dimension 1)
2. Find `k` = minimum of all `length[pos, i]` for `i = 1..n`
3. Count `p` = number of dimensions where `length[pos, i] == help` (the value of dim 1)
4. If `k == help` (all dimensions have the same length index):
   - `level = k * n + n - p`
   - Since all are equal, `p == n`, so `level = k * n + 0 = k * n`
   - Actually `level = k * n + n - p`. When all equal, `help == k` so `p` counts all
     dims with value `help`, which is `n`. Result: `k*n + n - n = k*n`.
5. If `k < help` (mixed lengths):
   - `level = k * n + p`
   - `p` = count of dims with length equal to `help` (the first dim's value, NOT the min)

**Note:** The `p` computation counts dimensions matching `help` (dim 1's value), not `k`
(the minimum). This is a subtle but critical detail. When lengths are mixed, `p` counts
how many dimensions have the *largest* side-length index (smallest side length), which
corresponds to dimension 1's value when it happens to be the max.

**When `jones == 1` (Jones Original / DIRECT):**
Simply returns the minimum side-length index across all dimensions:
- `level = min(length[pos, i]) for i = 1..n`

### Edge Cases
- 1D (`n=1`): `jones==0` returns `k*1 + 1 - 1 = k`, same as `jones==1`.
- Uniform lengths `[k,k,...,k]`: Both methods return `k*n` (Gablonsky) or `k` (Jones).

### Rust Mapping
→ `RectangleStorage::get_level()` in `storage.rs`

---

## 2. `direct_dirchoose_()` — Potentially Optimal Rectangle Selection

### Signature
```c
void direct_dirchoose_(integer *anchor, integer *s, integer *actdeep,
    doublereal *f, doublereal *minf, doublereal epsrel, doublereal epsabs,
    doublereal *thirds, integer *maxpos, integer *length, integer *maxfunc,
    const integer *maxdeep, const integer *maxdiv, integer *n, FILE *logfile,
    integer *cheat, doublereal *kmax, integer *ifeasiblef, integer jones)
```

### Parameters
| Parameter | Direction | Description |
|-----------|-----------|-------------|
| `anchor` | in | `[MAXDEEP+2]` head pointers for depth-sorted linked lists (shifted: `anchor[-1]` = infeasible) |
| `s` | out | `[MAXDIV × 2]` selected rectangles: `s[j,1]` = rect index, `s[j,2]` = level |
| `actdeep` | in | Current maximum depth |
| `f` | in | `[MAXFUNC × 2]` function values and feasibility |
| `minf` | in | Current minimum function value |
| `epsrel` | in | Relative epsilon for the epsilon test |
| `epsabs` | in | Absolute epsilon for the epsilon test |
| `thirds` | in | Precomputed `1/3^k` values |
| `maxpos` | out | Number of selected rectangles |
| `length` | in | Side-length indices array |
| `maxfunc` | in | Leading dimension |
| `maxdeep` | in | Maximum depth |
| `maxdiv` | in | Leading dimension of `s` |
| `n` | in | Problem dimension |
| `logfile` | in | Optional log file |
| `cheat` | in | If 1, cap helplower at `kmax` |
| `kmax` | in | Maximum slope cap for cheat mode |
| `ifeasiblef` | in | 1 if no feasible point has been found yet |
| `jones` | in | Algorithm method flag |

### Behavior

**Phase 1: Collect candidates (one per depth level)**

If `ifeasiblef >= 1` (no feasible point found yet):
- Scan depth levels `j = 0..actdeep`, find the first anchor with a rectangle.
- Select only that single rectangle and return immediately (`maxpos = 1`).

Otherwise:
- For each depth level `j = 0..actdeep`, if `anchor[j] > 0`, add the anchor head
  (the rectangle with the lowest f-value at that depth) to the candidate set `s`.
- Compute each candidate's level using `dirgetlevel_()`.
- Also check `anchor[-1]` for the infeasible list — store as `novalue`.

**Phase 2: Convex hull filtering (sweep from maxpos down to 1)**

For each candidate `j` (processed in reverse order, from largest to smallest depth):
1. Compute `helplower` = minimum positive slope to all candidates at *smaller* depths (i < j):
   - `slope = (f[i] - f[j]) / (thirds[level_i] - thirds[level_j])`
   - Only considers feasible candidates (`f[i, 2] <= 1`)
   - If slope ≤ 0, the candidate `j` is immediately eliminated (goto L60)
2. Compute `helpgreater` = maximum positive slope to all candidates at *larger* depths (i > j):
   - Same slope formula
   - If slope ≤ 0, candidate `j` is immediately eliminated
3. If `helpgreater > helplower`: candidate is above the convex hull → eliminate
4. **Epsilon test**: If `helpgreater <= helplower`:
   - If cheat mode: cap helplower at kmax
   - Test: `f[j] - helplower * thirds[level_j] > min(minf - epsrel*|minf|, minf - epsabs)`
   - If true, the candidate does not improve enough → eliminate

**Phase 3: Append infeasible candidate**

If `novalue > 0`, append the infeasible head as an extra selected rectangle.

### Key Observations
- The convex hull is computed by pairwise slope comparison, not monotone chain.
- Candidates are anchor heads only (lowest f-value rectangle at each depth).
- Elimination works backwards (largest depth first).
- The epsilon test uses `min(epsrel, epsabs)` thresholds — both relative AND absolute.
- When `s[j,1]` is set to 0, the candidate is eliminated.
- The f-value array uses `f[(idx << 1) + 1]` for value and `f[(idx << 1) + 2]` for feasibility.

### Rust Mapping
→ `PotentiallyOptimal::select()` in `storage.rs`

---

## 3. `direct_dirdoubleinsert_()` — Jones Original Equal-Value Insertion

### Signature
```c
void direct_dirdoubleinsert_(integer *anchor, integer *s, integer *maxpos,
    integer *point, doublereal *f, const integer *maxdeep,
    integer *maxfunc, const integer *maxdiv, integer *ierror)
```

### Parameters
| Parameter | Direction | Description |
|-----------|-----------|-------------|
| `anchor` | in | Depth-indexed anchor list heads |
| `s` | in/out | Selected rectangles array, may grow with appended entries |
| `maxpos` | in/out | Number of selected rectangles (may increase) |
| `point` | in | Linked-list next pointers |
| `f` | in | Function values array |
| `maxdeep` | in | Maximum depth (unused) |
| `maxfunc` | in | Leading dimension (unused) |
| `maxdiv` | in | Capacity of `s` array |
| `ierror` | out | Set to -6 if `s` overflows |

### Behavior

Only used for Jones Original DIRECT (algmethod=0). For each selected rectangle `i`:

1. Get the depth level from `s[i, 2]`.
2. Start at the anchor head `help = anchor[depth]`.
3. Walk the linked list via `point[]` starting from `pos = point[help]`.
4. While `pos > 0` and the f-value difference `f[pos] - f[help] <= 1e-13`:
   - Append `pos` to the selected set `s` with the same depth.
   - Advance `pos = point[pos]`.
5. Stop when f-value difference exceeds `1e-13` (tolerance for "equal value").
6. If `maxpos >= maxdiv`, set `ierror = -6` (overflow).

### Key Observations
- The tolerance for "equal value" is hardcoded at `1e-13`.
- Only checks rectangles at the SAME depth level (since the linked list is per-level).
- Only the original `oldmaxpos` entries are examined (not newly appended ones).
- Walks from the anchor head forward through the sorted list.

### Rust Mapping
→ `PotentiallyOptimal::double_insert()` in `storage.rs`

---

## 4. `direct_dirgetmaxdeep_()` — Maximum Depth (Minimum Length Index)

### Signature
```c
integer direct_dirgetmaxdeep_(integer *pos, integer *length, integer *maxfunc,
                               integer *n)
```

### Parameters
| Parameter | Direction | Description |
|-----------|-----------|-------------|
| `pos` | in | Rectangle position (1-based) |
| `length` | in | Side-length index array |
| `maxfunc` | in | Leading dimension (unused) |
| `n` | in | Problem dimension |

### Behavior

Returns the minimum side-length index across all `n` dimensions for rectangle at `pos`:
```
result = min(length[pos, i]) for i = 1..n
```

This is the "depth" of the rectangle — the smallest side-length index means the longest
remaining side, which corresponds to how deeply the rectangle has been divided.

### Key Observations
- Identical to `dirgetlevel_()` when `jones == 1`.
- The name "max deep" is confusing: it returns the minimum length index, which corresponds
  to the maximum (coarsest) remaining side length.
- Used to compute the `currentlength` parameter for `dirdivide_()`.

### Rust Mapping
→ `RectangleStorage::get_max_deep()` in `storage.rs`

---

## 5. `direct_dirget_i__()` — Find Longest Dimensions

### Signature
```c
void direct_dirget_i__(integer *length, integer *pos, integer *arrayi,
                       integer *maxi, integer *n, integer *maxfunc)
```

### Parameters
| Parameter | Direction | Description |
|-----------|-----------|-------------|
| `length` | in | Side-length index array |
| `pos` | in | Rectangle position (1-based) |
| `arrayi` | out | `[n]` array of dimension indices that have the minimum length index |
| `maxi` | out | Number of longest dimensions found |
| `n` | in | Problem dimension |
| `maxfunc` | in | Leading dimension (unused) |

### Behavior

1. Find `help` = minimum of `length[pos, i]` for `i = 1..n`.
   (Minimum length index = longest side = coarsest division.)
2. Collect all dimension indices `i` where `length[pos, i] == help` into `arrayi`.
3. Set `maxi` = count of such dimensions.

### Key Observations
- For a cube (all sides equal), `maxi = n` and all dimensions are returned.
- For a non-cube, only the longest (least-divided) dimensions are returned.
- This determines which dimensions will be trisected in `dirdivide_()`.
- `arrayi` is 1-based (dimensions 1..n).

### Rust Mapping
→ `RectangleStorage::get_longest_dims()` in `storage.rs`

---

## 6. `direct_dirsamplepoints_()` — Create New Sample Points

### Signature
```c
void direct_dirsamplepoints_(doublereal *c__, integer *arrayi,
    doublereal *delta, integer *sample, integer *start, integer *length,
    FILE *logfile, doublereal *f, integer *free,
    integer *maxi, integer *point, doublereal *x, doublereal *l,
    doublereal *minf, integer *minpos, doublereal *u, integer *n,
    integer *maxfunc, const integer *maxdeep, integer *oops)
```

### Parameters
| Parameter | Direction | Description |
|-----------|-----------|-------------|
| `c__` | in/out | `[n × MAXFUNC]` center coordinates |
| `arrayi` | in | Dimension indices to divide along (from `dirget_i__`) |
| `delta` | in | Offset magnitude = `thirds[currentlength + 1]` |
| `sample` | in | Parent rectangle index |
| `start` | out | First new rectangle index (head of chain) |
| `length` | in/out | Copied from parent to all new rects |
| `logfile` | in | Optional log file |
| `f` | unused | (declared but not accessed) |
| `free` | in/out | Head of free list; advanced as slots are consumed |
| `maxi` | in | Number of dimensions to divide |
| `point` | in/out | Linked-list pointers |
| `x, l, minf, minpos, u` | unused | (declared but not accessed) |
| `n` | in | Problem dimension |
| `maxfunc, maxdeep` | unused | Dimension parameters |
| `oops` | out | Set to 1 if free list runs out |

### Behavior

1. Allocate `2 * maxi` new rectangle slots from the free list.
2. For each new slot:
   - Copy ALL length indices from parent `sample`.
   - Copy ALL center coordinates from parent `sample`.
   - Advance the free pointer: `free = point[free]`.
3. Terminate the chain: `point[last_slot] = 0`.
4. Then offset the centers:
   - For each dimension `j` in `arrayi[1..maxi]`:
     - First rect in pair: `center[arrayi[j]] = parent_center[arrayi[j]] + delta`
     - Second rect in pair: `center[arrayi[j]] = parent_center[arrayi[j]] - delta`
5. The new rects are chained via `point[]` in allocation order.

### Key Observations
- Creates exactly `2 * maxi` new points (one positive offset, one negative, per dimension).
- New rects start as exact copies of the parent (all dims), then only the divided
  dimension is offset.
- `start` points to the first new rectangle; they are linked: start → point[start] → ...
- The function does NOT evaluate the objective — that's done by `dirsamplef_()`.
- If the free list is exhausted (`free == 0`), sets `oops = 1` and returns.

### Rust Mapping
→ `Direct::sample_points()` in `direct.rs`

---

## 7. `direct_dirdivide_()` — Rectangle Division (Trisection)

### Signature
```c
void direct_dirdivide_(integer *new__, integer *currentlength,
    integer *length, integer *point, integer *arrayi, integer *sample,
    integer *list2, doublereal *w, integer *maxi, doublereal *f,
    integer *maxfunc, const integer *maxdeep, integer *n)
```

### Parameters
| Parameter | Direction | Description |
|-----------|-----------|-------------|
| `new__` | in | Head of chain of new sample points (from `dirsamplepoints_`) |
| `currentlength` | in | Current minimum length index of the parent |
| `length` | in/out | Side-length index array — updated for divided dims |
| `point` | in | Linked-list pointers for new rect chain |
| `arrayi` | in | Dimension indices being divided |
| `sample` | in | Parent rectangle index |
| `list2` | scratch | `[n × 2]` temporary sorted list |
| `w` | scratch | `[n]` temporary min-f-value per dimension |
| `maxi` | in | Number of dimensions being divided |
| `f` | in | Function values |
| `maxfunc, maxdeep` | unused | Dimension parameters |
| `n` | in | Problem dimension |

### Behavior

**Phase 1: Compute w[j] = min(f+, f-) for each divided dimension**

Walk the chain of new points in pairs (positive offset, negative offset):
```
for i = 1..maxi:
    j = arrayi[i]
    w[j] = f[pos_positive]       // f-value at center + delta along dim j
    w[j] = min(w[j], f[pos_negative])  // min of + and - offsets
    Insert dim j into sorted list (list2) keyed by w[j]
```

**Phase 2: Sort dimensions by w[j] using `dirinsertlist_2__()`**

`dirinsertlist_2__()` is a linked-list insertion sort. Dimensions with smaller `min(f+, f-)`
are divided first (get a shorter side length = higher length index).

**Phase 3: Trisect in sorted order**

For each dimension in sorted order (smallest w first):
1. Increment `length[dim, sample]` (parent) to `currentlength + 1`.
2. For all remaining children in the chain (including current pair's children):
   - Increment `length[dim, child]` to `currentlength + 1`.

The critical insight: dimensions divided first get their length index incremented in ALL
remaining rectangles (parent + subsequent children). Dimensions divided last only get
incremented in the parent and their own two children. This means:
- Dims with smallest `min(f+, f-)` are divided first → become shorter first.
- All subsequent children also get shortened in that dimension.
- After all divisions, the parent and each pair of children have their lengths correctly
  reflecting which dimensions have been trisected.

### Key Observations
- The parent (`sample`) retains its center coordinates; only children have offset centers.
- After division, the parent's length indices are updated for ALL divided dimensions.
- Each child pair (pos+, pos-) gets length indices updated for their dimension AND all
  previously-divided dimensions.
- `dirinsertlist_2__()` and `dirsearchmin_()` are private helper functions that implement
  a sorted linked list for dimension ordering.
- `currentlength` is the value returned by `dirgetmaxdeep_()` before division.

### Rust Mapping
→ `Direct::divide_rectangle()` in `direct.rs`

---

## 8. `direct_dirinsertlist_()` — Insert Children into Anchor Lists

### Signature
```c
void direct_dirinsertlist_(integer *new__, integer *anchor, integer *point,
    doublereal *f, integer *maxi, integer *length, integer *maxfunc,
    const integer *maxdeep, integer *n, integer *samp, integer jones)
```

### Parameters
| Parameter | Direction | Description |
|-----------|-----------|-------------|
| `new__` | in/out | Head of chain of new rects; advanced as pairs are consumed |
| `anchor` | in/out | Depth-indexed anchor list heads |
| `point` | in/out | Linked-list next pointers |
| `f` | in | Function values |
| `maxi` | in | Number of dimension pairs (2*maxi new rects total) |
| `length` | in | Side-length index array |
| `maxfunc` | in | Leading dimension |
| `maxdeep` | in | Maximum depth (unused) |
| `n` | in | Problem dimension |
| `samp` | in | Parent rectangle index |
| `jones` | in | Algorithm method flag for level computation |

### Behavior

**For each of the `maxi` dimension pairs:**

1. Take two consecutive rects from the chain: `pos1 = new__`, `pos2 = point[pos1]`.
2. Advance `new__ = point[pos2]`.
3. Compute `deep = dirgetlevel_(pos1, ...)` for the pair's depth level.
4. Insert `pos1` and `pos2` into `anchor[deep]` linked list, maintaining sorted order
   by f-value. The logic handles 6 cases:
   - Empty list (`anchor[deep] == 0`): whichever has lower f becomes the anchor.
   - Non-empty list, multiple orderings of f(pos1), f(pos2), f(anchor):
     - If both are below current anchor: new anchor, careful ordering of all three.
     - If one is below: new anchor, insert other.
     - If neither is below: insert both using `dirinsert_()`.

**Then insert the parent `samp`:**

5. Compute `deep = dirgetlevel_(samp, ...)`.
6. If `f[samp] < f[anchor[deep]]`: `samp` becomes new anchor head.
7. Otherwise: insert via `dirinsert_()`.

### Key Observations
- `dirinsert_()` is a private helper that walks the linked list from a start position
  and inserts the new element in f-value sorted order.
- The function processes `2*maxi + 1` rectangles total (maxi pairs + parent).
- After insertion, each anchor list is sorted by ascending f-value.
- The level computation determines which linked list to insert into.
- Subtle bug fix (noted in comments): JG 08/30/00 fixed sorting when
  `f(pos2) < f(pos1) < f(anchor)`.

### Rust Mapping
→ `RectangleStorage::insert_into_list()` in `storage.rs`

---

## 9. `direct_dirgetlevel_()` — (Covered in Section 1 above)

---

## 10. `direct_dirgetmaxdeep_()` — (Covered in Section 4 above)

---

## 11. `direct_dirreplaceinf_()` — Infeasible Point Replacement

### Signature
```c
void direct_dirreplaceinf_(integer *free, integer *freeold,
    doublereal *f, doublereal *c__, doublereal *thirds, integer *length,
    integer *anchor, integer *point, doublereal *c1, doublereal *c2,
    integer *maxfunc, const integer *maxdeep, integer *maxdim, integer *n,
    FILE *logfile, doublereal *fmax, integer jones)
```

### Parameters
| Parameter | Direction | Description |
|-----------|-----------|-------------|
| `free` | in | Current free list head (used as upper bound for iteration) |
| `freeold` | in | Previous free list head (unused) |
| `f` | in/out | Function values — infeasible entries may be updated |
| `c__` | in | Center coordinates |
| `thirds` | in | Precomputed `1/3^k` values |
| `length` | in | Side-length index array |
| `anchor` | in/out | Anchor lists (may be re-sorted) |
| `point` | in/out | Linked-list pointers (may be re-sorted) |
| `c1, c2` | in | Scaling coefficients (`xs1`, `xs2` from `dirpreprc_`) |
| `maxfunc` | in | Leading dimension |
| `maxdeep` | in | Maximum depth |
| `maxdim` | in | Maximum dimensions |
| `n` | in | Problem dimension |
| `logfile` | in | Optional log file |
| `fmax` | in | Maximum feasible function value found so far |
| `jones` | in | Algorithm method flag |

### Behavior

For each rectangle `i = 1..free-1` that is infeasible (`f[i, 2] > 0`):

1. **Compute bounding box**: For each dimension `j`, compute:
   - `sidelength = thirds[length[i, j]]`
   - `a[j] = center[i, j] - sidelength`
   - `b[j] = center[i, j] + sidelength`

2. **Reset f-value**: `f[i, 1] = HUGE_VAL`, `f[i, 2] = 2.0` (mark as fully infeasible).

3. **Search for nearby feasible points**: For each rectangle `k = 1..free-1`:
   - If `k` is feasible (`f[k, 2] == 0`):
     - Check if `k`'s center is inside the bounding box `[a, b]` using `isinbox_()`.
     - If inside: `f[i, 1] = min(f[i, 1], f[k, 1])`, set `f[i, 2] = 1.0`
       (mark as "replaced infeasible").

4. **After scanning all feasible points**:
   - If `f[i, 2] == 1` (a nearby feasible point was found):
     - Add perturbation: `f[i, 1] += |f[i, 1]| * 1e-6`
     - Call `dirresortlist_()` to re-sort the anchor list containing rectangle `i`.
   - If `f[i, 2] != 1` (no nearby feasible point):
     - Set `f[i, 1] = max(fmax + 1, f[i, 1])` (give it a high value).

### `isinbox_()` helper (static)
Returns 1 if all coordinates of `x` are within `[a, b]`, 0 otherwise.

### `dirresortlist_()` helper (static)
1. Compute the level of the rectangle to find the correct anchor list.
2. Remove the rectangle from its current position in the linked list.
3. Re-insert it at the correct sorted position based on its new f-value.
4. If the new value is lower than the current anchor, become the new anchor.

### Key Observations
- Infeasible points are identified by `f[i, 2] > 0`.
- Feasibility flags: `0` = feasible, `1` = replaced infeasible, `2` = infeasible.
- The bounding box uses the actual side lengths (via `thirds[]`), NOT a fixed growth factor.
  The comment about `sidelength = thirds[help] * 2` on line 566 is from `dirgetmaxdeep_`
  but is immediately overwritten per-dimension.
- The perturbation `1e-6 * |f|` ensures replaced infeasible points sort slightly higher
  than the feasible point they borrowed a value from.
- After replacement, the linked list is re-sorted to maintain the invariant.
- This function is called once per main-loop iteration, after all selected rects are divided.
- The `c1[l], c2[l]` scaling computation on line 620-621 appears to compute unscaled
  coordinates but the result is stored in `x[]` and not used — this may be dead code from
  the original Fortran.

### Rust Mapping
→ `RectangleStorage::replace_infeasible()` in `storage.rs`

---

## 12. `direct_dirinfcn_()` — Unscale, Evaluate, Rescale

### Signature
```c
void direct_dirinfcn_(fp fcn, doublereal *x, doublereal *c1,
    doublereal *c2, integer *n, doublereal *f, integer *flag__,
    void *fcn_data)
```

### Parameters
| Parameter | Direction | Description |
|-----------|-----------|-------------|
| `fcn` | in | User objective function pointer |
| `x` | in/out | Point in normalized [0,1]^n space; unscaled then rescaled |
| `c1` | in | Scale factors `xs1[i] = u[i] - l[i]` |
| `c2` | in | Offset factors `xs2[i] = l[i] / (u[i] - l[i])` |
| `n` | in | Problem dimension |
| `f` | out | Function value at the unscaled point |
| `flag__` | out | Feasibility flag from user function (0=feasible, nonzero=infeasible) |
| `fcn_data` | in | Opaque user data |

### Behavior

1. **Unscale**: `x[i] = (x[i] + c2[i]) * c1[i]` for each dimension.
   - This maps from normalized `[0,1]^n` to actual bounds `[l, u]`.
   - Algebraically: `(x_norm + l/(u-l)) * (u-l) = x_norm*(u-l) + l`.
   - When `x_norm = 0.5` (center): `0.5*(u-l) + l = (u+l)/2` (midpoint).

2. **Evaluate**: `*f = fcn(n, x, flag, fcn_data)`.
   - User function sets `flag` to indicate feasibility.

3. **Rescale**: `x[i] = x[i] / c1[i] - c2[i]` for each dimension.
   - Inverse of the unscaling: `x_actual / (u-l) - l/(u-l) = (x_actual - l)/(u-l)`.

### Key Observations
- The `x` array is modified in-place (unscaled for the call, then rescaled back).
- This is thread-unsafe in the original C code — each call modifies shared center coords.
  For the Rust port, use a separate scratch buffer.
- `flag__` is initialized to 0 before calling fcn. The user function can set it nonzero
  to indicate infeasibility.
- The NLOPT wrapper `direct_objective_func` signature: `double fcn(int n, const double *x, int *undefined_flag, void *data)`.

### Rust Mapping
→ `Direct::evaluate()` + `to_actual()` in `direct.rs`

---

## 13. `direct_dirpreprc_()` — Preprocessing: Bounds Scaling

### Signature
```c
void direct_dirpreprc_(doublereal *u, doublereal *l, integer *n,
                       doublereal *xs1, doublereal *xs2, integer *oops)
```

### Parameters
| Parameter | Direction | Description |
|-----------|-----------|-------------|
| `u` | in | Upper bounds array |
| `l` | in | Lower bounds array |
| `n` | in | Problem dimension |
| `xs1` | out | Scale factors: `xs1[i] = u[i] - l[i]` |
| `xs2` | out | Offset factors: `xs2[i] = l[i] / (u[i] - l[i])` |
| `oops` | out | 1 if any `u[i] <= l[i]`, 0 otherwise |

### Behavior

1. **Validation pass**: Check `u[i] > l[i]` for all `i`. If any violates, set `oops = 1` and return.
2. **Scaling computation**: For each dimension `i`:
   - `help = u[i] - l[i]` (range width)
   - `xs2[i] = l[i] / help` (normalized lower bound offset)
   - `xs1[i] = help` (range width = scale factor)

### Algebraic Properties
- `to_actual(x_norm) = (x_norm + xs2) * xs1 = x_norm * (u-l) + l`
- `to_normalized(x_actual) = x_actual / xs1 - xs2 = (x_actual - l) / (u-l)`
- `to_actual(0) = l`, `to_actual(1) = u`, `to_actual(0.5) = (l+u)/2`
- Roundtrip: `to_normalized(to_actual(x)) = x` exactly (no floating-point error for exact inputs)

### Key Observations
- Simple affine transformation, no special cases needed.
- In `DIRect.c`, the bounds arrays `l` and `u` are overwritten with `xs1` and `xs2`
  respectively (then restored from `oldl`/`oldu` at the end).

### Rust Mapping
→ `Direct::new()` scale/offset computation in `direct.rs`

---

## 14. `direct_dirheader_()` — Input Validation and Logging

### Signature
```c
void direct_dirheader_(FILE *logfile, integer *version,
    doublereal *x, integer *n, doublereal *eps, integer *maxf, integer *maxt,
    doublereal *l, doublereal *u, integer *algmethod, integer *maxfunc,
    const integer *maxdeep, doublereal *fglobal, doublereal *fglper,
    integer *ierror, doublereal *epsfix, integer *iepschange,
    doublereal *volper, doublereal *sigmaper)
```

### Behavior

1. **Epsilon handling**:
   - If `eps < 0`: set `iepschange = 1`, `epsfix = |eps|`, `eps = |eps|`.
     (Negative eps triggers Jones dynamic epsilon update formula.)
   - If `eps >= 0`: set `iepschange = 0`, `epsfix = 1e100` (effectively disabled).

2. **Logging**: Print algorithm version, dimension, parameters to logfile.

3. **Bounds validation**: For each dimension, check `u[i] > l[i]`.
   - If violated: set `ierror = -1`, count errors.

4. **Capacity check**: If `maxf + 20 > maxfunc`, warn and set `ierror = -2`.

### Key Observations
- The epsilon sign convention is the user interface for selecting dynamic vs fixed epsilon.
- This function is called once at the beginning of `direct_direct_()`.
- `ierror` is only set negative; the caller checks and returns early if negative.

### Rust Mapping
→ `Direct::validate_inputs()` in `direct.rs`

---

## 15. `direct_dirinit_()` — Initialization (First Iteration)

### Signature
```c
void direct_dirinit_(doublereal *f, fp fcn, doublereal *c__,
    integer *length, integer *actdeep, integer *point, integer *anchor,
    integer *free, FILE *logfile, integer *arrayi,
    integer *maxi, integer *list2, doublereal *w, doublereal *x,
    doublereal *l, doublereal *u, doublereal *minf, integer *minpos,
    doublereal *thirds, doublereal *levels, integer *maxfunc,
    const integer *maxdeep, integer *n, integer *maxor, doublereal *fmax,
    integer *ifeasiblef, integer *iinfeasible, integer *ierror,
    void *fcndata, integer jones, double starttime, double maxtime,
    int *force_stop)
```

### Behavior

**Step 1: Precompute thirds[] and levels[]**

For `jones == 0` (Gablonsky):
- `w[j+1] = sqrt(n - j + j/9) * 0.5` for `j = 0..n-1` (distance from center to corner
  for rectangles with `j` shorter sides)
- `levels[(i-1)*n + j] = w[j+1] / 3^i` for `i = 1..maxdeep/n`, `j = 0..n-1`

For `jones == 1` (Jones Original):
- `levels[i] = 1/3^i` for `i = 1..maxdeep`
- `levels[0] = 1`

For both:
- `thirds[i] = 1/3^i` for `i = 1..maxdeep`
- `thirds[0] = 1`

**Step 2: Evaluate center point**

- Set center = `(0.5, 0.5, ..., 0.5)` for all `n` dimensions.
- Set all length indices to 0 (unit cube).
- Call `dirinfcn_()` to unscale and evaluate the objective function.
- Handle feasibility: if infeasible (`flag > 0`), set `f[1,1] = HUGE_VAL`, `ifeasiblef = 1`.
- Set `minf = f[1,1]`, `minpos = 1`, `actdeep = 2`, `free = 2`.

**Step 3: Sample, evaluate, and divide**

- `delta = thirds[1] = 1/3`
- Call `dirget_i__()` to get longest dimensions of rect 1 (all dims, since it's a cube).
- Call `dirsamplepoints_()` to create `2*n` new points.
- Call `dirsamplef_()` to evaluate all `2*n` new points.
- Call `dirdivide_()` to sort dimensions and trisect.
- Call `dirinsertlist_()` to insert all `2*n + 1` rectangles into anchor lists.

**Total function evaluations after init: `2*n + 1`.**

### Key Observations
- The levels[] array for Gablonsky (`jones == 0`) is 2D: indexed by `(iteration, offset)`,
  while for Jones (`jones == 1`) it's 1D.
- The w[] computation `sqrt(n - j + j/9)` models the Euclidean distance from center to
  corner of a rectangle with `j` sides of length `1/3^(k+1)` and `n-j` sides of length `1/3^k`.
  The `j/9` term accounts for the shorter sides contributing less to the diagonal.
- `actdeep = 2` after init (not 1) because the initial division creates depth-1 and depth-2
  rectangles.
- Error codes: `-4` = sample points allocation failed, `-5` = sample evaluation failed,
  `-102` = force_stop triggered, `DIRECT_MAXTIME_EXCEEDED` = time limit.

### Rust Mapping
→ `Direct::initialize()` in `direct.rs`

---

## 16. `direct_dirinitlist_()` — Initialize Linked Lists

### Signature
```c
void direct_dirinitlist_(integer *anchor, integer *free, integer *point,
                         doublereal *f, integer *maxfunc, const integer *maxdeep)
```

### Behavior

1. Set all anchors to 0 (empty lists): `anchor[i] = 0` for `i = -1..maxdeep`.
   (The `++anchor;` adjustment enables `anchor[-1]` for the infeasible list.)

2. Initialize all f-values to 0: `f[i, 1] = 0, f[i, 2] = 0` for `i = 1..maxfunc`.

3. Set up the free list: `point[i] = i + 1` for `i = 1..maxfunc-1`, `point[maxfunc] = 0`.
   This creates a singly-linked list `1 → 2 → 3 → ... → maxfunc → 0`.

4. Set `free = 1` (first free position).

### Key Observations
- Called once during initialization by `direct_direct_()`.
- The free list provides O(1) allocation of new rectangle slots.
- Anchor index `-1` is for infeasible rectangles (a special linked list).

### Rust Mapping
→ `RectangleStorage::init_lists()` in `storage.rs`

---

## 17. `direct_dirsummary_()` — Final Summary Logging

### Signature
```c
void direct_dirsummary_(FILE *logfile, doublereal *x, doublereal *l,
    doublereal *u, integer *n, doublereal *minf, doublereal *fglobal,
    integer *numfunc, integer *ierror)
```

### Behavior
Prints final results to log file: function value, evaluation count, optimality gap, and
the final solution vector with distances to bounds. Purely diagnostic, no algorithmic logic.

### Rust Mapping
→ `Direct::summary()` in `direct.rs`

---

## Private Helper Functions

### `dirinsert_()` — Sorted Linked-List Insertion
Walks from `*start` through the linked list and inserts `*ins` at the correct position
to maintain ascending f-value order. If `f[ins] < f[point[start]]`, inserts immediately;
otherwise advances. If reaching end of list (`point[start] == 0`), appends.

### `dirinsertlist_2__()` — Dimension Sort Insertion
Used by `dirdivide_()` to sort dimensions by their `w[j]` values (min of f+ and f-).
Maintains a linked list sorted by `w[]` values. Each entry also stores the corresponding
rectangle position `k` in column 2 of `list2`.

### `dirsearchmin_()` — Extract Minimum from Sorted List
Pops the head of the sorted dimension list (created by `dirinsertlist_2__`), returning
the dimension index `k` and rectangle position `pos`.

### `isinbox_()` — Point-in-Box Test
Returns 1 if all coordinates of `x` are within `[a, b]`, 0 otherwise. Used by
`dirreplaceinf_()` to check proximity of feasible points to infeasible ones.

### `dirresortlist_()` — Re-sort After Infeasible Replacement
Removes a rectangle from its anchor list, then re-inserts it at the correct sorted
position based on its updated f-value. Called by `dirreplaceinf_()` after an infeasible
point's f-value has been replaced.

---

## Summary: Function Call Graph

```
direct_direct_() [DIRect.c]
├── direct_dirheader_()       — validate inputs, handle epsilon
├── direct_dirinitlist_()     — initialize linked lists and free list
├── direct_dirpreprc_()       — compute scaling coefficients
├── direct_dirinit_()         — first iteration
│   ├── direct_dirinfcn_()    — evaluate center point
│   ├── direct_dirget_i__()   — get longest dimensions (all for cube)
│   ├── direct_dirsamplepoints_() — create 2n new points
│   ├── direct_dirsamplef_()  — evaluate all new points [DIRserial.c]
│   ├── direct_dirdivide_()   — sort dims, trisect
│   │   ├── dirinsertlist_2__() — sort dimensions by min(f+,f-)
│   │   └── dirsearchmin_()    — pop minimum dimension
│   └── direct_dirinsertlist_() — insert into anchor lists
│       └── dirinsert_()       — sorted list insertion
├── [main loop]
│   ├── direct_dirchoose_()   — select potentially optimal rects
│   ├── direct_dirdoubleinsert_() — add equal-value rects (Jones only)
│   ├── direct_dirgetmaxdeep_() — get depth of selected rect
│   ├── direct_dirget_i__()   — get longest dimensions
│   ├── direct_dirsamplepoints_() — create new sample points
│   ├── direct_dirsamplef_()  — evaluate new points
│   ├── direct_dirdivide_()   — sort dims, trisect
│   ├── direct_dirinsertlist_() — insert children
│   ├── direct_dirreplaceinf_() — handle infeasible points
│   │   ├── direct_dirgetmaxdeep_() — bounding box computation
│   │   ├── isinbox_()         — proximity test
│   │   └── dirresortlist_()   — re-sort after replacement
│   │       └── direct_dirgetlevel_() — level for list lookup
│   └── direct_dirgetlevel_() — level computation (various uses)
└── direct_dirsummary_()      — final logging
```
