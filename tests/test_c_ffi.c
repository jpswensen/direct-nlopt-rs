/* test_c_ffi.c — C test program that calls the Rust DIRECT library via FFI.
 *
 * Compiles and links against the Rust static library (libdirect_nlopt.a).
 * Verifies the FFI interface produces correct optimization results and
 * compares with NLOPT C results when available.
 *
 * Build (after cargo build --release):
 *   cc -O2 -o test_c_ffi tests/test_c_ffi.c \
 *     -L target/release -ldirect_nlopt -lm -lpthread -ldl
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Include the generated Rust FFI header */
#include "../include/direct_nlopt.h"

/* NLOPT return codes (matching direct.h) */
#define DIRECT_INVALID_BOUNDS      (-1)
#define DIRECT_MAXFEVAL_EXCEEDED    1
#define DIRECT_MAXITER_EXCEEDED     2
#define DIRECT_GLOBAL_FOUND         3
#define DIRECT_VOLTOL               4
#define DIRECT_SIGMATOL             5

/* ──────────────────────────────────────────────────────────────────────────── */
/* Test objective functions                                                    */
/* ──────────────────────────────────────────────────────────────────────────── */

static double sphere(int n, const double *x, int *undefined_flag, void *data) {
    (void)undefined_flag;
    (void)data;
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += x[i] * x[i];
    return sum;
}

static double rosenbrock(int n, const double *x, int *undefined_flag, void *data) {
    (void)undefined_flag;
    (void)data;
    double sum = 0.0;
    for (int i = 0; i < n - 1; i++) {
        double t1 = x[i + 1] - x[i] * x[i];
        double t2 = 1.0 - x[i];
        sum += 100.0 * t1 * t1 + t2 * t2;
    }
    return sum;
}

static double rastrigin(int n, const double *x, int *undefined_flag, void *data) {
    (void)undefined_flag;
    (void)data;
    double sum = 10.0 * n;
    for (int i = 0; i < n; i++)
        sum += x[i] * x[i] - 10.0 * cos(2.0 * M_PI * x[i]);
    return sum;
}

/* ──────────────────────────────────────────────────────────────────────────── */
/* Test helpers                                                                */
/* ──────────────────────────────────────────────────────────────────────────── */

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_TRUE(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
        tests_failed++; \
        return 1; \
    } else { \
        tests_passed++; \
    } \
} while(0)

/* ──────────────────────────────────────────────────────────────────────────── */
/* Tests                                                                       */
/* ──────────────────────────────────────────────────────────────────────────── */

static int test_sphere_gablonsky(void) {
    printf("test_sphere_gablonsky...\n");
    double lb[] = {-5.0, -5.0};
    double ub[] = {5.0, 5.0};
    double x[2] = {0.0, 0.0};
    double minf = 1e30;

    int ret = direct_nlopt_optimize(
        sphere, NULL,
        2, lb, ub, x, &minf,
        500, 0,
        0.0, 0.0,    /* start, maxtime */
        1e-4, 0.0,   /* magic_eps, magic_eps_abs */
        0.0, -1.0,   /* volume_reltol, sigma_reltol */
        NULL,         /* force_stop */
        -HUGE_VAL, 0.0, /* fglobal, fglobal_reltol */
        NULL,         /* logfile */
        DIRECT_NLOPT_ALGORITHM_GABLONSKY
    );

    ASSERT_TRUE(ret > 0, "return code should be positive (success)");
    ASSERT_TRUE(minf < 1e-4, "minf should be < 1e-4");
    ASSERT_TRUE(fabs(x[0]) < 0.1, "x[0] should be near 0");
    ASSERT_TRUE(fabs(x[1]) < 0.1, "x[1] should be near 0");
    printf("  OK: ret=%d, minf=%.6e, x=[%.6f, %.6f]\n", ret, minf, x[0], x[1]);
    return 0;
}

static int test_sphere_original(void) {
    printf("test_sphere_original...\n");
    double lb[] = {-5.0, -5.0};
    double ub[] = {5.0, 5.0};
    double x[2] = {0.0, 0.0};
    double minf = 1e30;

    int ret = direct_nlopt_optimize(
        sphere, NULL,
        2, lb, ub, x, &minf,
        500, 0,
        0.0, 0.0,
        1e-4, 0.0,
        0.0, -1.0,
        NULL,
        -HUGE_VAL, 0.0,
        NULL,
        DIRECT_NLOPT_ALGORITHM_ORIGINAL
    );

    ASSERT_TRUE(ret > 0, "return code should be positive (success)");
    ASSERT_TRUE(minf < 1e-2, "minf should be < 1e-2");
    printf("  OK: ret=%d, minf=%.6e, x=[%.6f, %.6f]\n", ret, minf, x[0], x[1]);
    return 0;
}

static int test_rosenbrock(void) {
    printf("test_rosenbrock...\n");
    double lb[] = {-5.0, -5.0};
    double ub[] = {5.0, 5.0};
    double x[2] = {0.0, 0.0};
    double minf = 1e30;

    int ret = direct_nlopt_optimize(
        rosenbrock, NULL,
        2, lb, ub, x, &minf,
        2000, 0,
        0.0, 0.0,
        1e-4, 0.0,
        0.0, -1.0,
        NULL,
        -HUGE_VAL, 0.0,
        NULL,
        DIRECT_NLOPT_ALGORITHM_GABLONSKY
    );

    ASSERT_TRUE(ret > 0, "return code should be positive (success)");
    ASSERT_TRUE(minf < 5.0, "minf should be < 5.0");
    printf("  OK: ret=%d, minf=%.6e, x=[%.6f, %.6f]\n", ret, minf, x[0], x[1]);
    return 0;
}

static int test_rastrigin(void) {
    printf("test_rastrigin...\n");
    double lb[] = {-5.12, -5.12};
    double ub[] = {5.12, 5.12};
    double x[2] = {0.0, 0.0};
    double minf = 1e30;

    int ret = direct_nlopt_optimize(
        rastrigin, NULL,
        2, lb, ub, x, &minf,
        2000, 0,
        0.0, 0.0,
        1e-4, 0.0,
        0.0, -1.0,
        NULL,
        -HUGE_VAL, 0.0,
        NULL,
        DIRECT_NLOPT_ALGORITHM_GABLONSKY
    );

    ASSERT_TRUE(ret > 0, "return code should be positive (success)");
    ASSERT_TRUE(minf < 5.0, "minf should be < 5.0 for rastrigin");
    printf("  OK: ret=%d, minf=%.6e, x=[%.6f, %.6f]\n", ret, minf, x[0], x[1]);
    return 0;
}

static int test_optimize_full(void) {
    printf("test_optimize_full...\n");
    double lb[] = {-5.0, -5.0};
    double ub[] = {5.0, 5.0};
    double x[2] = {0.0, 0.0};
    double minf = 1e30;

    struct direct_nlopt_result result = direct_nlopt_optimize_full(
        sphere, NULL,
        2, lb, ub, x, &minf,
        500, 0,
        1e-4, 0.0,
        0.0, -1.0,
        NULL,
        -HUGE_VAL, 0.0,
        DIRECT_NLOPT_ALGORITHM_GABLONSKY
    );

    ASSERT_TRUE(result.return_code > 0, "return_code should be positive");
    ASSERT_TRUE(result.nfev > 0, "nfev should be > 0");
    ASSERT_TRUE(result.nit > 0, "nit should be > 0");
    ASSERT_TRUE(minf < 1e-4, "minf should be < 1e-4");
    printf("  OK: ret=%d, nfev=%d, nit=%d, minf=%.6e\n",
           result.return_code, result.nfev, result.nit, minf);
    return 0;
}

static int test_higher_dim(void) {
    printf("test_higher_dim (5D sphere)...\n");
    double lb[] = {-5.0, -5.0, -5.0, -5.0, -5.0};
    double ub[] = {5.0, 5.0, 5.0, 5.0, 5.0};
    double x[5] = {0};
    double minf = 1e30;

    int ret = direct_nlopt_optimize(
        sphere, NULL,
        5, lb, ub, x, &minf,
        5000, 0,
        0.0, 0.0,
        1e-4, 0.0,
        0.0, -1.0,
        NULL,
        -HUGE_VAL, 0.0,
        NULL,
        DIRECT_NLOPT_ALGORITHM_GABLONSKY
    );

    ASSERT_TRUE(ret > 0, "return code should be positive (success)");
    ASSERT_TRUE(minf < 1.0, "minf should be < 1.0 for 5D sphere");
    printf("  OK: ret=%d, minf=%.6e\n", ret, minf);
    return 0;
}

static int test_version(void) {
    printf("test_version...\n");
    const char *ver = direct_nlopt_version();
    ASSERT_TRUE(ver != NULL, "version should not be NULL");
    ASSERT_TRUE(strlen(ver) > 0, "version should not be empty");
    printf("  OK: version=%s\n", ver);
    return 0;
}

static int test_invalid_dimension(void) {
    printf("test_invalid_dimension...\n");
    double lb[] = {-5.0};
    double ub[] = {5.0};
    double x[1] = {0.0};
    double minf = 1e30;

    int ret = direct_nlopt_optimize(
        sphere, NULL,
        0, lb, ub, x, &minf,
        500, 0,
        0.0, 0.0,
        1e-4, 0.0,
        0.0, -1.0,
        NULL,
        -HUGE_VAL, 0.0,
        NULL,
        DIRECT_NLOPT_ALGORITHM_GABLONSKY
    );

    ASSERT_TRUE(ret < 0, "return code should be negative (error)");
    printf("  OK: ret=%d (error as expected)\n", ret);
    return 0;
}

/* ──────────────────────────────────────────────────────────────────────────── */
/* Main                                                                        */
/* ──────────────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("=== direct-nlopt-rs C FFI Test Suite ===\n\n");

    test_sphere_gablonsky();
    test_sphere_original();
    test_rosenbrock();
    test_rastrigin();
    test_optimize_full();
    test_higher_dim();
    test_version();
    test_invalid_dimension();

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           tests_passed, tests_run, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
