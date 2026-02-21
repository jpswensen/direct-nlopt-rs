/* Minimal utility functions needed by NLOPT DIRECT code.
   Provides only nlopt_stop_time_() and nlopt_seconds() which are the
   only nlopt-util functions used by the DIRECT algorithm files. */

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

int nlopt_stop_time_(double start, double maxtime)
{
    return (maxtime > 0 && nlopt_seconds() - start >= maxtime);
}

/* nlopt_isinf, nlopt_isnan - not used by DIRECT but provided for completeness
   if needed during linking */
int nlopt_isinf(double x) { return isinf(x); }
int nlopt_isnan(double x) { return isnan(x); }
int nlopt_isfinite(double x) { return isfinite(x); }
int nlopt_istiny(double x) { return fabs(x) < 1e-300; }

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
