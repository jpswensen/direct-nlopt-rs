/*
 * nlopt_bench.c — C benchmark program for NLOPT DIRECT algorithm.
 *
 * Compiles and links against the NLOPT DIRECT C source files to time
 * direct_optimize() for standard test functions, matching the Rust
 * Criterion benchmarks for apples-to-apples comparison.
 *
 * Compile:
 *   cc -O3 -o nlopt_bench benchmarks/nlopt_bench.c \
 *      nlopt/src/algs/direct/DIRect.c \
 *      nlopt/src/algs/direct/DIRsubrout.c \
 *      nlopt/src/algs/direct/DIRserial.c \
 *      nlopt/src/algs/direct/direct_wrap.c \
 *      nlopt-shim/nlopt_util_shim.c \
 *      -I nlopt-shim -I nlopt/src/algs/direct \
 *      -lm
 *
 * Output format (JSON-like, one line per benchmark):
 *   {"name":"sphere_2d_gablonsky","dim":2,"algo":"GABLONSKY","maxfeval":5000,
 *    "time_ms":1.234,"nfev":500,"nit":25,"minf":1.23e-10,
 *    "x":[0.001,-0.002]}
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "direct.h"

/* ── Test functions ── */

static double sphere(int n, const double *x, int *undefined_flag, void *data) {
    (void)undefined_flag; (void)data;
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    return sum;
}

static double rosenbrock(int n, const double *x, int *undefined_flag, void *data) {
    (void)undefined_flag; (void)data;
    double sum = 0.0;
    for (int i = 0; i < n - 1; i++) {
        double t1 = x[i + 1] - x[i] * x[i];
        double t2 = 1.0 - x[i];
        sum += 100.0 * t1 * t1 + t2 * t2;
    }
    return sum;
}

static double rastrigin(int n, const double *x, int *undefined_flag, void *data) {
    (void)undefined_flag; (void)data;
    double sum = 10.0 * n;
    for (int i = 0; i < n; i++)
        sum += x[i] * x[i] - 10.0 * cos(2.0 * M_PI * x[i]);
    return sum;
}

/* ── Timing utility ── */

static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ── Benchmark runner ── */

typedef struct {
    const char *name;
    direct_objective_func func;
    int dim;
    double lb;
    double ub;
    int maxfeval;
    direct_algorithm algo;
    const char *algo_name;
} benchmark_config;

static void run_benchmark(const benchmark_config *cfg, int num_runs) {
    double *lower = (double *)malloc(cfg->dim * sizeof(double));
    double *upper = (double *)malloc(cfg->dim * sizeof(double));
    double *x     = (double *)malloc(cfg->dim * sizeof(double));

    for (int i = 0; i < cfg->dim; i++) {
        lower[i] = cfg->lb;
        upper[i] = cfg->ub;
    }

    /* Warmup run */
    double minf;
    direct_optimize(cfg->func, NULL,
                    cfg->dim, lower, upper,
                    x, &minf,
                    cfg->maxfeval, -1,
                    0.0, 0.0,
                    1e-4, 0.0,
                    0.0, -1.0,
                    NULL,
                    -HUGE_VAL, 0.0,
                    NULL,
                    cfg->algo);

    /* Timed runs */
    double total_ms = 0.0;
    int last_nfev = 0;
    int last_nit = 0;
    double last_minf = 0.0;

    for (int run = 0; run < num_runs; run++) {
        memset(x, 0, cfg->dim * sizeof(double));

        double t0 = get_time_ms();
        direct_return_code rc = direct_optimize(
            cfg->func, NULL,
            cfg->dim, lower, upper,
            x, &minf,
            cfg->maxfeval, -1,
            0.0, 0.0,
            1e-4, 0.0,
            0.0, -1.0,
            NULL,
            -HUGE_VAL, 0.0,
            NULL,
            cfg->algo);
        double t1 = get_time_ms();

        total_ms += (t1 - t0);
        last_minf = minf;
        (void)rc;
    }

    double avg_ms = total_ms / num_runs;

    /* Print JSON result */
    printf("{\"name\":\"%s\",\"dim\":%d,\"algo\":\"%s\",\"maxfeval\":%d,"
           "\"time_ms\":%.3f,\"minf\":%.15e,\"x\":[",
           cfg->name, cfg->dim, cfg->algo_name, cfg->maxfeval,
           avg_ms, last_minf);
    for (int i = 0; i < cfg->dim; i++) {
        if (i > 0) printf(",");
        printf("%.15e", x[i]);
    }
    printf("]}\n");

    free(lower);
    free(upper);
    free(x);
}

int main(int argc, char **argv) {
    int num_runs = 10;
    if (argc > 1) num_runs = atoi(argv[1]);
    if (num_runs < 1) num_runs = 1;

    fprintf(stderr, "NLOPT C Benchmark — %d runs per config\n", num_runs);

    benchmark_config configs[] = {
        /* Sphere */
        {"sphere_2d_gablonsky",  sphere,  2, -5.0,  5.0,  5000, DIRECT_GABLONSKY, "GABLONSKY"},
        {"sphere_2d_original",   sphere,  2, -5.0,  5.0,  5000, DIRECT_ORIGINAL,  "ORIGINAL"},
        {"sphere_5d_gablonsky",  sphere,  5, -5.0,  5.0,  5000, DIRECT_GABLONSKY, "GABLONSKY"},
        {"sphere_5d_original",   sphere,  5, -5.0,  5.0,  5000, DIRECT_ORIGINAL,  "ORIGINAL"},
        {"sphere_10d_gablonsky", sphere, 10, -5.0,  5.0,  5000, DIRECT_GABLONSKY, "GABLONSKY"},
        {"sphere_10d_original",  sphere, 10, -5.0,  5.0,  5000, DIRECT_ORIGINAL,  "ORIGINAL"},

        /* Rosenbrock */
        {"rosenbrock_2d_gablonsky", rosenbrock, 2, -5.0, 5.0, 5000, DIRECT_GABLONSKY, "GABLONSKY"},
        {"rosenbrock_2d_original",  rosenbrock, 2, -5.0, 5.0, 5000, DIRECT_ORIGINAL,  "ORIGINAL"},
        {"rosenbrock_5d_gablonsky", rosenbrock, 5, -5.0, 5.0, 5000, DIRECT_GABLONSKY, "GABLONSKY"},
        {"rosenbrock_5d_original",  rosenbrock, 5, -5.0, 5.0, 5000, DIRECT_ORIGINAL,  "ORIGINAL"},

        /* Rastrigin */
        {"rastrigin_2d_gablonsky", rastrigin, 2, -5.12, 5.12, 5000, DIRECT_GABLONSKY, "GABLONSKY"},
        {"rastrigin_2d_original",  rastrigin, 2, -5.12, 5.12, 5000, DIRECT_ORIGINAL,  "ORIGINAL"},
        {"rastrigin_5d_gablonsky", rastrigin, 5, -5.12, 5.12, 5000, DIRECT_GABLONSKY, "GABLONSKY"},
        {"rastrigin_5d_original",  rastrigin, 5, -5.12, 5.12, 5000, DIRECT_ORIGINAL,  "ORIGINAL"},
    };

    int n_configs = sizeof(configs) / sizeof(configs[0]);
    for (int i = 0; i < n_configs; i++) {
        run_benchmark(&configs[i], num_runs);
    }

    return 0;
}
