# NLOPT Gablonsky DIRECT Implementation Analysis

## Source File: `nlopt/src/algs/direct/DIRect.c`

The file is a Fortran-to-C translation (via f2c, hand-cleaned by SGJ) of Gablonsky's
DIRECT implementation. It contains a single function: `direct_direct_()`.

---

## 1. Entry Point: `direct_direct_()`

### Signature

```c
void direct_direct_(
    fp fcn,              // Objective function pointer
    doublereal *x,       // [n] Output: best point found
    integer *n,          // Dimension of the problem
    doublereal *eps,     // Epsilon parameter (Jones magic eps); <0 triggers update formula
    doublereal epsabs,   // Absolute epsilon parameter
    integer *maxf,       // Max function evaluations (in/out: returns actual count)
    integer *maxt,       // Max iterations (in/out)
    double starttime,    // Start time for maxtime check
    double maxtime,      // Maximum wall-clock time
    int *force_stop,     // External stop flag (set by caller)
    doublereal *minf,    // Output: minimum function value found
    doublereal *l,       // [n] Lower bounds (modified during run, restored at end)
    doublereal *u,       // [n] Upper bounds (modified during run, restored at end)
    integer *algmethod,  // 0=DIRECT_ORIGINAL (Jones), 1=DIRECT_GABLONSKY (DIRECT-L)
    integer *ierror,     // Output: return code (negative=error, positive=termination)
    FILE *logfile,       // Optional log file (NULL to suppress)
    doublereal *fglobal, // Known global minimum (DIRECT_UNKNOWN_FGLOBAL if unknown)
    doublereal *fglper,  // Percent error tolerance for fglobal termination
    doublereal *volper,  // Volume tolerance (percentage)
    doublereal *sigmaper,// Sigma/measure tolerance
    void *fcn_data       // Opaque user data passed to fcn
);
```

### Key Observations
- **1-based indexing**: The Fortran heritage means `--u; --l; --x;` pointer adjustments
  are applied at the start, so all array access uses 1-based indices internally.
- The `l` and `u` arrays are **modified in-place** by `dirpreprc_()` (overwritten with
  scaling coefficients), then restored from `oldl`/`oldu` at the end.
- `algmethod` is stored in local variable `jones` and used throughout.

---

## 2. Variable Map

### Dynamically Allocated Arrays

| Variable | Size | Indexing | Purpose |
|----------|------|----------|---------|
| `c__` | `MAXFUNC × n` | `c__[i + pos*n - n - 1]` (1-based) | Center coordinates of each rectangle. Row `pos` holds the n-dimensional center in normalized [0,1]^n space. |
| `f` | `MAXFUNC × 2` | `f[(pos<<1) - 2]` = f-value, `f[(pos<<1) - 1]` = feasibility flag | Function values and feasibility. Flag: 0=feasible, 2=infeasible. |
| `length` | `MAXFUNC × n` | `length[i + pos*n - n - 1]` | Side-length indices per dimension per rectangle. Value `k` means side length = `thirds[k]` = (1/3)^k. |
| `point` | `MAXFUNC` | `point[pos - 1]` | Linked-list next pointers. Also serves as the free list. |
| `anchor` | `MAXDEEP + 2` | `anchor[depth + 1]` (depth 0..MAXDEEP, +1 for infeasible list at index 0) | Head pointers for linked lists keyed by depth/level. `anchor[-1]` (C index 0) is the infeasible list. |
| `s` | `MAXDIV × 2` | `s[j-1]` = rect index, `s[j + MAXDIV - 1]` = depth | Selected (potentially optimal) rectangles from `dirchoose_()`. |
| `thirds` | `MAXDEEP + 1` | `thirds[k]` | Precomputed values: `thirds[k] = (1/3)^k`. |
| `levels` | `MAXDEEP + 1` | `levels[k]` | Precomputed level/measure values used for sigma tolerance. |
| `w` | `n` | `w[j]` | Workspace for `dirdivide_()`: stores min(f+, f-) per dimension. |
| `oldl` | `n` | `oldl[i-1]` | Saved original lower bounds (restored at end). |
| `oldu` | `n` | `oldu[i-1]` | Saved original upper bounds (restored at end). |
| `list2` | `n × 2` | Used by `dirdivide_()` | Workspace for dimension sorting during division. |
| `arrayi` | `n` | `arrayi[j]` | Indices of dimensions with maximum (longest) side length. |

### Scalar Variables

| Variable | Type | Purpose |
|----------|------|---------|
| `jones` | integer | Copy of `*algmethod`: 0=Original, 1=Gablonsky |
| `ifree` | integer | Head of free list (next available rectangle slot) |
| `minpos` | integer | Index (1-based) of rectangle with minimum f-value |
| `*minf` | doublereal | Current minimum function value found |
| `fmax` | doublereal | Maximum feasible function value found (for infeasible replacement) |
| `numfunc` | integer | Total number of function evaluations so far |
| `actdeep` | integer | Current depth being processed |
| `actmaxdeep` | integer | Maximum depth reached so far |
| `actdeep_div__` | integer | Depth of current rectangle being divided |
| `maxi` | integer | Number of longest dimensions in current rectangle |
| `maxpos` | integer | Number of selected rectangles from `dirchoose_()` |
| `newtosample` | integer | Count of new sample points in current iteration |
| `delta` | doublereal | Distance from center to new sample points = `thirds[depth+1]` |
| `ifeasiblef` | integer | 0=feasible point found, 1=no feasible point yet |
| `iinfesiblef` | integer | >0 if any infeasible point encountered |
| `iepschange` | integer | 1=use Jones epsilon update formula, 0=fixed epsilon |
| `epsfix` | doublereal | Absolute value of epsilon (used when eps<0 for Jones formula) |
| `divfactor` | doublereal | `max(1, |fglobal|)` for fglobal termination check |
| `increase` | integer | Flag: 1=budget was increased because no feasible point found |
| `oldmaxf` | integer | Original maxf budget before any increases |
| `ifreeold` | integer | Previous free list head, for `dirreplaceinf_()` |
| `version` | integer | 204 (version counter, unused in logic) |
| `cheat` | integer | 0 (obsolete flag, always 0) |
| `kmax` | doublereal | 1e10 (obsolete cap, effectively unused) |

---

## 3. Memory Allocation Scheme

```
MAXFUNC = maxf <= 0 ? 101000 : (maxf + 1000 + maxf/2)
MAXDEEP = maxt <= 0 ? MAXFUNC/5 : (maxt + 1000)
MAXDIV  = 5000  (constant — max selected rectangles per iteration)
```

**Rationale**: MAXFUNC is 1.5× the requested budget + 1000 as headroom, because
a single iteration may evaluate many more points than the remaining budget allows
(evaluations are checked after each iteration, not after each evaluation).

**Layout Note (SGJ)**: The original Fortran stored arrays as `length(MAXFUNC, n)`
(column-major), meaning dimension was the contiguous direction. SGJ transposed to
`length[MAXFUNC][n]` (row-major) so that adding new rectangles means adding
contiguous rows, making realloc possible without data movement.

Allocated arrays and their sizes:
- `c__`: `MAXFUNC × n` doubles
- `length`: `MAXFUNC × n` integers
- `f`: `MAXFUNC × 2` doubles
- `point`: `MAXFUNC` integers
- `s`: `MAXDIV × 2` integers
- `anchor`: `MAXDEEP + 2` integers
- `levels`: `MAXDEEP + 1` doubles
- `thirds`: `MAXDEEP + 1` doubles
- `w`: `n` doubles
- `oldl`, `oldu`: `n` doubles each
- `list2`: `n × 2` integers
- `arrayi`: `n` integers

---

## 4. Initialization Sequence

### Step 1: `direct_dirheader_()`
**Purpose**: Validate inputs and set up epsilon handling.

- Checks `l[i] < u[i]` for all dimensions → `ierror = -1` if violated
- Checks `MAXFUNC >= maxf` → `ierror = -2` if violated
- Handles epsilon sign:
  - `eps > 0` → `iepschange = 0` (fixed epsilon)
  - `eps < 0` → `iepschange = 1` (Jones update), `epsfix = |eps|`, `eps = |eps|`
- Validates `volper`, `sigmaper`

### Step 2: `divfactor` computation
```c
if (fglobal == 0.0)
    divfactor = 1.0;
else
    divfactor = fabs(fglobal);
```
Used in fglobal termination: `(minf - fglobal) * 100 / divfactor <= fglper`.

### Step 3: `direct_dirinitlist_()`
**Purpose**: Initialize linked-list data structures.

- Sets all `anchor[i] = 0` for `i = -1..MAXDEEP` (using 1-based offset)
- Sets up `point[]` as a free list: `point[i] = i+2` for `i=0..MAXFUNC-2`,
  `point[MAXFUNC-1] = 0` (sentinel)
- Sets `free = 1` (first free slot, 1-based)
- Initializes `f[i] = 0` for all positions

### Step 4: `direct_dirpreprc_()`
**Purpose**: Compute scaling coefficients to map bounds to unit cube.

- For each dimension `i`:
  - `xs1[i] = u[i] - l[i]` (scale factor)
  - `xs2[i] = l[i] / (u[i] - l[i])` (offset)
- **Critical**: `l` and `u` are overwritten in-place with `xs1` and `xs2` respectively.
  This means `l[i] = u_orig[i] - l_orig[i]` and `u[i] = l_orig[i] / (u_orig[i] - l_orig[i])`
  after this call.
- Validation: checks `u[i] - l[i] > 0`, sets `oops = 1` if violated.
- Reverse mapping: `x_actual = (x_normalized + xs2) * xs1 = (x_norm + l/(u-l)) * (u-l) = x_norm*(u-l) + l`

### Step 5: `direct_dirinit_()`
**Purpose**: Perform the first iteration of DIRECT.

1. Precompute `thirds[k] = (1/3)^k` for `k = 0..MAXDEEP`
2. Precompute `levels[k]` based on algorithm variant
3. Evaluate the center point `(0.5, ..., 0.5)` of the unit cube
4. For each dimension `d = 0..n-1`:
   - Sample at `center ± thirds[1]` (= center ± 1/3) along dimension `d`
5. Evaluate all `2n` sample points
6. Divide the initial rectangle:
   - Sort dimensions by `min(f+, f-)` for each dimension
   - Trisect in sorted order (best dimensions first get smallest resulting rectangles)
7. Insert resulting rectangles into linked lists
8. Total function evaluations after init = `2n + 1`

### Post-init setup:
```c
numfunc = maxi + 1 + maxi;  // = 2*n + 1 (since maxi = n for first rect)
actmaxdeep = 1;
oldpos = 0;
tstart = 2;  // main loop starts at iteration 2
```

---

## 5. Main Iteration Loop

```
for (t = tstart; t <= maxt; ++t) {
```

### 5a. SELECT: `direct_dirchoose_()`

```c
actdeep = actmaxdeep;
direct_dirchoose_(anchor, s, &MAXDEEP, f, minf, eps, epsabs, levels,
                  &maxpos, length, &MAXFUNC, &MAXDEEP, &MAXDIV, n,
                  logfile, &cheat, &kmax, &ifeasiblef, jones);
```

- Identifies potentially optimal (PO) rectangles using convex hull analysis
- Output: `s[j-1]` = rectangle index (1-based), `s[j+MAXDIV-1]` = depth level
- `maxpos` = number of selected rectangles
- See `dirchoose_()` analysis in DIRsubrout.c analysis for full details

### 5b. DOUBLE INSERT (Original only): `direct_dirdoubleinsert_()`

```c
if (algmethod == 0) {
    direct_dirdoubleinsert_(anchor, s, &maxpos, point, f, &MAXDEEP,
                            &MAXFUNC, &MAXDIV, ierror);
}
```

- Only for DIRECT_ORIGINAL (Jones): adds to selection all rectangles at the
  same level with the same function value as already-selected rectangles
- Can increase `maxpos`
- Error `-6` if `maxpos` exceeds `MAXDIV`

### 5c. Process Each Selected Rectangle

```c
for (j = 1; j <= maxpos; ++j) {
    actdeep = s[j + MAXDIV - 1];
    if (s[j - 1] > 0) {  // skip if index is 0 (removed)
```

For each selected rectangle with index `help = s[j-1]`:

#### 5c.i. Compute depth and delta
```c
actdeep_div__ = direct_dirgetmaxdeep_(&help, length, &MAXFUNC, n);
delta = thirds[actdeep_div__ + 1];
```
- `actdeep_div__` = minimum of all `length[help][d]` values (= depth of the rectangle)
- `delta` = `(1/3)^(depth+1)` = half the new sub-rectangle side length

#### 5c.ii. Check depth limit
```c
if (actdeep + 1 >= mdeep) {
    ierror = -6;
    goto L100;
}
actmaxdeep = max(actdeep, actmaxdeep);
```

#### 5c.iii. Remove from linked list
```c
// Remove 'help' from the linked list anchored at anchor[actdeep + 1]
if (anchor[actdeep + 1] == help) {
    anchor[actdeep + 1] = point[help - 1];  // was head → advance head
} else {
    // Walk list to find predecessor, splice out 'help'
    pos1 = anchor[actdeep + 1];
    while (point[pos1 - 1] != help)
        pos1 = point[pos1 - 1];
    point[pos1 - 1] = point[help - 1];
}
```
- If `actdeep < 0`, this is an infeasible rectangle; recover actual depth from `f[(help<<1)-2]`

#### 5c.iv. Get longest dimensions
```c
direct_dirget_i__(length, &help, arrayi, &maxi, n, &MAXFUNC);
```
- Finds all dimensions with minimum `length` value (= longest side length, since
  smaller index = longer side)
- `maxi` = count of such dimensions, `arrayi[0..maxi-1]` = their indices

#### 5c.v. Create sample points
```c
direct_dirsamplepoints_(c__, arrayi, &delta, &help, &start, length,
                         logfile, f, &ifree, &maxi, point, x, l,
                         minf, &minpos, u, n, &MAXFUNC, &MAXDEEP, &oops);
```
- Creates `2 × maxi` new sample points by copying parent center and offsetting
  by `±delta` along each longest dimension
- Allocates slots from the free list
- `start` = first new slot index

#### 5c.vi. Evaluate sample points
```c
direct_dirsamplef_(c__, arrayi, &delta, &help, &start, length,
                    logfile, f, &ifree, &maxi, point, fcn, x, l,
                    minf, &minpos, u, n, &MAXFUNC, &MAXDEEP, &oops,
                    &fmax, &ifeasiblef, &iinfesiblef, fcn_data, force_stop);
```
- Evaluates objective function at each new point (in serial: `DIRserial.c`)
- Updates `minf`, `minpos`, `fmax`, feasibility flags
- Checks `force_stop` and `maxtime` after evaluation

#### 5c.vii. Divide the rectangle
```c
direct_dirdivide_(&start, &actdeep_div__, length, point, arrayi,
                   &help, list2, w, &maxi, f, &MAXFUNC, &MAXDEEP, n);
```
- Sorts dimensions by `min(f+, f-)` — dimensions with better samples get divided first
- Trisects along each longest dimension in sorted order
- Dimensions divided first end up with higher `length` indices (smaller rectangles)

#### 5c.viii. Insert into sorted lists
```c
direct_dirinsertlist_(&start, anchor, point, f, &maxi, length,
                       &MAXFUNC, &MAXDEEP, n, &help, jones);
```
- Inserts parent + all children into the appropriate linked list (sorted by f-value)
- Level computed by `dirgetlevel_()` which differs between Original and Gablonsky

#### 5c.ix. Update function count
```c
numfunc += maxi + maxi;  // = 2 * maxi new evaluations
```

### 5d. Termination Checks (after processing all selected rectangles)

The checks are performed in this specific order:

#### Check 1: Volume tolerance
```c
jones_save = jones; jones = 0;  // temporarily force Original level computation
actdeep_div__ = direct_dirgetlevel_(&minpos, length, &MAXFUNC, n, jones);
jones = jones_save;
delta = thirds[actdeep_div__] * 100;  // as percentage
if (delta <= volper) { ierror = 4; goto L100; }
```
Note: Volume check always uses `jones=0` (Original) level computation regardless
of algorithm variant.

#### Check 2: Sigma (measure) tolerance
```c
actdeep_div__ = direct_dirgetlevel_(&minpos, length, &MAXFUNC, n, jones);
delta = levels[actdeep_div__];
if (delta <= sigmaper) { ierror = 5; goto L100; }
```
Note: Sigma check uses the actual `jones` value, so level computation matches
the algorithm variant.

#### Check 3: Known global minimum
```c
if ((minf - fglobal) * 100 / divfactor <= fglper) { ierror = 3; goto L100; }
```

#### Check 4: Infeasible point replacement
```c
if (iinfesiblef > 0) {
    direct_dirreplaceinf_(&ifree, &ifreeold, f, c__, thirds, length,
                           anchor, point, u, l, &MAXFUNC, &MAXDEEP,
                           n, n, logfile, &fmax, jones);
}
ifreeold = ifree;
```

#### Check 5: Jones epsilon update
```c
if (iepschange == 1) {
    eps = max(|minf| * 1e-4, epsfix);
}
```

#### Check 6: Budget increase for infeasible problems
```c
if (increase == 1) {
    maxf = numfunc + oldmaxf;
    if (ifeasiblef == 0) increase = 0;
}
```

#### Check 7: Max function evaluations
```c
if (numfunc > maxf) {
    if (ifeasiblef == 0) { ierror = 1; goto L100; }
    else { increase = 1; maxf = numfunc + oldmaxf; }
}
```
Note: If no feasible point found yet, budget is extended rather than terminating.

#### End of loop → Check 8: Max iterations
After the loop exits naturally: `ierror = 2` (max iterations exceeded).

### 5e. Finalization (label L100)

```c
// Extract best point, unscaling from normalized to actual coordinates
for (i = 1; i <= n; ++i) {
    x[i] = c__[i + minpos*n - n - 1] * l[i] + l[i] * u[i];
    // Restore original bounds
    u[i] = oldu[i-1];
    l[i] = oldl[i-1];
}
maxf = numfunc;  // return actual function evaluation count
direct_dirsummary_(logfile, x, l, u, n, minf, fglobal, &numfunc, ierror);
```

**Unscaling formula**: `x_actual = c_normalized * xs1 + xs1 * xs2`
= `c_norm * (u-l) + (u-l) * l/(u-l)` = `c_norm * (u-l) + l`

This is correct because in normalized space, `c ∈ [0,1]`, so
`x = c*(u-l) + l` maps back to `[l, u]`.

---

## 6. Epsilon Update Logic

The epsilon parameter controls the "Pareto" trade-off between rectangle size
and function value in the potentially-optimal selection.

### Fixed epsilon (`eps > 0` on input):
- `iepschange = 0`
- `eps` stays constant throughout

### Jones update formula (`eps < 0` on input):
- `iepschange = 1`
- `epsfix = |eps|`
- `eps = |eps|` initially
- After each iteration: `eps = max(|minf| * 1e-4, epsfix)`
- This makes epsilon adapt to the current best value scale

---

## 7. Control Flow Diagram (Pseudocode)

```
direct_direct_(fcn, x, n, eps, ...) {
    // === MEMORY ALLOCATION ===
    MAXFUNC = maxf + 1000 + maxf/2
    MAXDEEP = MAXFUNC/5 or maxt+1000
    Allocate c__[MAXFUNC×n], f[MAXFUNC×2], length[MAXFUNC×n],
             point[MAXFUNC], anchor[MAXDEEP+2], s[MAXDIV×2],
             thirds[MAXDEEP+1], levels[MAXDEEP+1], w[n], etc.

    // === INITIALIZATION ===
    oldl, oldu ← save original bounds
    dirheader_()          // validate inputs, set up epsilon
    divfactor ← max(1, |fglobal|)
    dirinitlist_()        // init linked lists and free list
    dirpreprc_(u, l)      // l,u overwritten with xs1,xs2 scaling
    dirinit_(...)         // first iteration: eval center + 2n pts, divide

    numfunc ← 2*n + 1
    actmaxdeep ← 1

    // === MAIN LOOP ===
    for t = 2 to maxt:
        // SELECT potentially optimal rectangles
        dirchoose_(anchor, s, ...) → maxpos selected
        
        // DOUBLE INSERT (Original only)
        if algmethod == 0:
            dirdoubleinsert_(s, ...) → may increase maxpos
        
        // PROCESS each selected rectangle
        for j = 1 to maxpos:
            if s[j] > 0:
                depth ← dirgetmaxdeep_(s[j])
                delta ← thirds[depth + 1]
                
                Remove s[j] from anchor list
                dirget_i_() → arrayi[0..maxi-1] = longest dims
                dirsamplepoints_() → create 2*maxi new points
                dirsamplef_() → evaluate function at new points
                    if force_stop or maxtime: goto FINALIZE
                dirdivide_() → trisect along longest dims
                dirinsertlist_() → insert children into sorted lists
                numfunc += 2 * maxi
        
        // TERMINATION CHECKS (in order)
        1. Volume tolerance:  thirds[level]*100 <= volper?     → ierror=4
        2. Sigma tolerance:   levels[level] <= sigmaper?        → ierror=5
        3. Global found:      (minf-fglobal)*100/div <= fglper? → ierror=3
        4. Replace infeasible points if any exist
        5. Update epsilon if using Jones formula
        6. Handle budget increase for infeasible problems
        7. Max evaluations:   numfunc > maxf?                   → ierror=1
    
    // Loop exhausted → ierror = 2 (max iterations)
    
    FINALIZE:
        // Unscale best point: x = c_norm * xs1 + xs1 * xs2
        // Restore original bounds from oldl, oldu
        // Report maxf = numfunc
        // Print summary
        // Free all memory
}
```

---

## 8. Key Design Decisions and Quirks

### 8.1 Fortran Heritage
- All major arrays use 1-based indexing despite being C code
- The `--u; --l; --x;` pointer adjustments at the top shift pointers so `[1]` accesses
  the first element
- `anchor` is offset so `anchor[-1]` (index 0) holds the infeasible list head

### 8.2 Bounds Modification
- `l` and `u` are destructively overwritten by `dirpreprc_()` with scaling coefficients
- Original bounds are saved in `oldl`/`oldu` and restored at finalization
- This means during the algorithm, `l[i] = u_orig - l_orig` (scale) and
  `u[i] = l_orig / (u_orig - l_orig)` (offset)

### 8.3 Budget Elasticity
- If no feasible point is found, the budget is increased: `maxf = numfunc + oldmaxf`
- The `increase` flag tracks this state
- Budget checking happens after all selected rects are processed, not per-evaluation

### 8.4 Volume Check Uses jones=0
- The volume tolerance check temporarily sets `jones=0` regardless of algorithm
- This means it always uses the Original (Jones) level computation for volume
- The sigma tolerance check uses the actual algorithm's level computation

### 8.5 MAXDIV Limitation
- `MAXDIV = 5000` is hardcoded — limits the number of PO rects per iteration
- `dirdoubleinsert_()` can trigger error `-6` if this is exceeded
- Only affects DIRECT_ORIGINAL (jones=0)

### 8.6 Depth Check Placement
- Depth limit `actdeep + 1 >= mdeep` is checked inside the per-rectangle loop
- If triggered, the algorithm jumps to finalization (preserving results so far)
- `mdeep = MAXDEEP` (the allocated depth capacity)
