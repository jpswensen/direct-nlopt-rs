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
