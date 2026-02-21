/* Minimal utility functions needed by NLOPT DIRECT code.
   Base stopping/utility functions are provided by stop.c.
   This shim provides nlopt_seconds() (which stop.c calls but does not define)
   and additional helper functions for comparison testing. */

#include <math.h>
#include <float.h>

#ifdef __APPLE__
#include <sys/time.h>
#elif defined(HAVE_GETTIMEOFDAY) || defined(HAVE_SYS_TIME_H)
#include <sys/time.h>
#else
#include <time.h>
#endif

double nlopt_seconds(void)
{
    static _Thread_local int start_inited = 0;
#if defined(__APPLE__) || defined(HAVE_GETTIMEOFDAY)
    static _Thread_local struct timeval start;
    struct timeval tv;
    if (!start_inited) {
        start_inited = 1;
        gettimeofday(&start, NULL);
    }
    gettimeofday(&tv, NULL);
    return (tv.tv_sec - start.tv_sec) + 1.e-6 * (tv.tv_usec - start.tv_usec);
#else
    return (double)time(NULL);
#endif
}

/* Stub for nlopt_iurand — used by cdirect.c for randomized variants only.
   For which_alg=0 and which_alg=13 this is never reached; provide a
   deterministic implementation for safety. */
#include <stdlib.h>
int nlopt_iurand(int n)
{
    return n > 0 ? (rand() % n) : 0;
}

/* ─── Thirds/levels precomputation helpers for comparison testing ───
   These extract the exact loops from direct_dirinit_() in DIRsubrout.c
   so that tests can compare NLOPT C output with Rust without needing
   the full dirinit_ setup. */

void nlopt_shim_precompute_thirds(double *thirds, int maxdeep)
{
    double help2 = 3.0;
    int i;
    for (i = 1; i <= maxdeep; ++i) {
        thirds[i] = 1.0 / help2;
        help2 *= 3.0;
    }
    thirds[0] = 1.0;
}

/* ─── cdirect rect_diameter / convex_hull shims for comparison testing ───
   These replicate the exact algorithms from cdirect.c (SGJ re-implementation)
   so tests can compare NLOPT C output with Rust CDirect without compiling
   the full cdirect.c + redblack tree infrastructure. */

/* Matches rect_diameter() in cdirect.c lines 94–112.
   Rounds to float precision (performance hack for convex_hull grouping). */
double nlopt_shim_rect_diameter(int n, const double *w, int which_diam)
{
    int i;
    if (which_diam == 0) {
        /* Jones measure: Euclidean distance from center to vertex */
        double sum = 0;
        for (i = 0; i < n; ++i)
            sum += w[i] * w[i];
        return ((float)(sqrt(sum) * 0.5));
    } else {
        /* Gablonsky measure: half-width of longest side */
        double maxw = 0;
        for (i = 0; i < n; ++i)
            if (w[i] > maxw)
                maxw = w[i];
        return ((float)(maxw * 0.5));
    }
}

/* Matches convex_hull() in cdirect.c lines 261–378.
   Input: npts entries sorted by (diameter, f_value, age) in lexicographic order.
   diameters[i], f_values[i]: the (d, f) coordinates for each point.
   hull_indices[]: output indices of hull points (allocated by caller, size >= npts).
   allow_dups: if nonzero, include duplicate points at (xmin,yminmin) and (xmax,ymaxmin).
   Returns: number of hull points written to hull_indices. */
int nlopt_shim_convex_hull(int npts, const double *diameters, const double *f_values,
                           int *hull_indices, int allow_dups)
{
    int nhull = 0;
    int i, start_idx, nmax_start;
    double xmin, xmax, yminmin, ymaxmin, minslope;

    if (npts <= 0) return 0;

    xmin = diameters[0];
    yminmin = f_values[0];
    xmax = diameters[npts - 1];

    /* Include initial points at (xmin, yminmin) */
    if (allow_dups) {
        for (i = 0; i < npts; ++i) {
            if (diameters[i] == xmin && f_values[i] == yminmin)
                hull_indices[nhull++] = i;
            else
                break;
        }
    } else {
        hull_indices[nhull++] = 0;
    }

    if (xmin == xmax) return nhull;

    /* Find ymaxmin: minimum f among entries with d == xmax */
    ymaxmin = HUGE_VAL;
    nmax_start = npts;
    for (i = 0; i < npts; ++i) {
        if (diameters[i] == xmax) {
            if (nmax_start == npts) nmax_start = i;
            if (f_values[i] < ymaxmin)
                ymaxmin = f_values[i];
        }
    }

    minslope = (ymaxmin - yminmin) / (xmax - xmin);

    /* Skip entries with d == xmin */
    start_idx = 0;
    for (i = 0; i < npts; ++i) {
        if (diameters[i] != xmin) {
            start_idx = i;
            break;
        }
    }

    /* Process entries between xmin and xmax */
    for (i = start_idx; i < nmax_start; ++i) {
        double x = diameters[i];
        double y = f_values[i];

        /* Skip if above the line from (xmin,yminmin) to (xmax,ymaxmin) */
        if (y > yminmin + (x - xmin) * minslope)
            continue;

        /* Performance hack: skip vertical lines (same diameter) */
        if (nhull > 0 && x == diameters[hull_indices[nhull - 1]]) {
            if (y > f_values[hull_indices[nhull - 1]]) {
                /* Skip to next diameter value */
                double cur_d = x;
                while (i < nmax_start && diameters[i] == cur_d)
                    ++i;
                --i; /* compensate for loop increment */
                continue;
            } else {
                /* Equal y or lower y at same diameter */
                if (allow_dups) {
                    hull_indices[nhull++] = i;
                    continue;
                }
                /* If equal y and not allow_dups, fall through to hull update */
            }
        }

        /* Remove points until we make a "left turn" to point i */
        while (nhull > 1) {
            double t1_d = diameters[hull_indices[nhull - 1]];
            double t1_f = f_values[hull_indices[nhull - 1]];
            double t2_d, t2_f;
            int it2 = nhull - 2;

            /* Look backwards for a different point */
            while (it2 >= 0) {
                t2_d = diameters[hull_indices[it2]];
                t2_f = f_values[hull_indices[it2]];
                if (t2_d != t1_d || t2_f != t1_f)
                    break;
                --it2;
            }
            if (it2 < 0) break;
            t2_d = diameters[hull_indices[it2]];
            t2_f = f_values[hull_indices[it2]];

            /* Cross product (t1-t2) × (k-t2) >= 0 means left turn or straight */
            if ((t1_d - t2_d) * (y - t2_f) - (t1_f - t2_f) * (x - t2_d) >= 0)
                break;
            --nhull;
        }
        hull_indices[nhull++] = i;
    }

    /* Add points at (xmax, ymaxmin) */
    if (allow_dups) {
        for (i = nmax_start; i < npts; ++i) {
            if (diameters[i] == xmax && f_values[i] == ymaxmin)
                hull_indices[nhull++] = i;
            else if (diameters[i] != xmax)
                break;
        }
    } else {
        /* Find entry with min f at xmax */
        for (i = nmax_start; i < npts; ++i) {
            if (diameters[i] == xmax && f_values[i] == ymaxmin) {
                hull_indices[nhull++] = i;
                break;
            }
        }
    }

    return nhull;
}

void nlopt_shim_precompute_levels(double *levels, int maxdeep, int n, int jones)
{
    int i, j;
    double help2;

    if (jones == 0) {
        /* Jones Original: w[j] = sqrt(n - j + j/9) * 0.5 for j=0..n-1 */
        double w[128]; /* enough for any reasonable n */
        for (j = 0; j <= n - 1; ++j) {
            w[j] = sqrt((double)(n - j) + (double)j / 9.0) * 0.5;
        }
        help2 = 1.0;
        for (i = 1; i <= maxdeep / n; ++i) {
            for (j = 0; j <= n - 1; ++j) {
                levels[(i - 1) * n + j] = w[j] / help2;
            }
            help2 *= 3.0;
        }
    } else {
        /* Gablonsky: levels[k] = 1/3^k */
        help2 = 3.0;
        for (i = 1; i <= maxdeep; ++i) {
            levels[i] = 1.0 / help2;
            help2 *= 3.0;
        }
        levels[0] = 1.0;
    }
}
