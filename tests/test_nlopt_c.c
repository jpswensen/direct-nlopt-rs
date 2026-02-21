/* Test C program that calls NLOPT's direct_optimize() with known inputs.
   Compile with:
     cc -O2 -o test_nlopt_c tests/test_nlopt_c.c \
       ../nlopt/src/algs/direct/DIRect.c \
       ../nlopt/src/algs/direct/DIRsubrout.c \
       ../nlopt/src/algs/direct/DIRserial.c \
       ../nlopt/src/algs/direct/direct_wrap.c \
       ../nlopt/src/util/timer.c ../nlopt/src/util/stop.c \
       -I nlopt-shim -I ../nlopt/src/algs/direct -I ../nlopt/src/util -lm
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "direct.h"

/* Sphere function: f(x) = sum(x_i^2), global min at x=0, f=0 */
static double sphere(int n, const double *x, int *undefined_flag, void *data) {
    double sum = 0.0;
    int i;
    (void)undefined_flag;
    (void)data;
    for (i = 0; i < n; i++)
        sum += x[i] * x[i];
    return sum;
}

/* Rosenbrock function: global min at x=(1,...,1), f=0 */
static double rosenbrock(int n, const double *x, int *undefined_flag, void *data) {
    double sum = 0.0;
    int i;
    (void)undefined_flag;
    (void)data;
    for (i = 0; i < n - 1; i++) {
        double t1 = x[i + 1] - x[i] * x[i];
        double t2 = 1.0 - x[i];
        sum += 100.0 * t1 * t1 + t2 * t2;
    }
    return sum;
}

static void run_test(const char *name, direct_objective_func f,
                     int dim, const double *lb, const double *ub,
                     int max_feval, direct_algorithm alg) {
    double *x = (double *)calloc(dim, sizeof(double));
    double minf = 0.0;
    int force_stop = 0;
    const char *alg_name = (alg == DIRECT_ORIGINAL) ? "ORIGINAL" : "GABLONSKY";

    direct_return_code rc = direct_optimize(
        f, NULL,
        dim, lb, ub,
        x, &minf,
        max_feval, -1,       /* max_feval, max_iter */
        0.0, 0.0,            /* start time, max time */
        1e-4, 0.0,           /* magic_eps, magic_eps_abs */
        0.0, 0.0,            /* volume_reltol, sigma_reltol */
        &force_stop,
        DIRECT_UNKNOWN_FGLOBAL, DIRECT_UNKNOWN_FGLOBAL_RELTOL,
        NULL,                 /* logfile */
        alg
    );

    printf("TEST: %s (dim=%d, alg=%s, maxfeval=%d)\n", name, dim, alg_name, max_feval);
    printf("  return_code = %d\n", (int)rc);
    printf("  minf = %.15e\n", minf);
    printf("  x = [");
    for (int i = 0; i < dim; i++) {
        if (i > 0) printf(", ");
        printf("%.15e", x[i]);
    }
    printf("]\n\n");

    free(x);
}

int main(void) {
    printf("=== NLOPT DIRECT C Test Results ===\n\n");

    /* Sphere 2D tests */
    {
        double lb[] = {-5.0, -5.0};
        double ub[] = {5.0, 5.0};
        run_test("sphere", sphere, 2, lb, ub, 500, DIRECT_GABLONSKY);
        run_test("sphere", sphere, 2, lb, ub, 500, DIRECT_ORIGINAL);
    }

    /* Rosenbrock 2D tests */
    {
        double lb[] = {-5.0, -5.0};
        double ub[] = {5.0, 5.0};
        run_test("rosenbrock", rosenbrock, 2, lb, ub, 2000, DIRECT_GABLONSKY);
        run_test("rosenbrock", rosenbrock, 2, lb, ub, 2000, DIRECT_ORIGINAL);
    }

    /* Sphere 3D test */
    {
        double lb[] = {-5.0, -5.0, -5.0};
        double ub[] = {5.0, 5.0, 5.0};
        run_test("sphere_3d", sphere, 3, lb, ub, 1000, DIRECT_GABLONSKY);
    }

    /* Sphere 5D test */
    {
        double lb[] = {-5.0, -5.0, -5.0, -5.0, -5.0};
        double ub[] = {5.0, 5.0, 5.0, 5.0, 5.0};
        run_test("sphere_5d", sphere, 5, lb, ub, 2000, DIRECT_GABLONSKY);
    }

    printf("=== All tests complete ===\n");
    return 0;
}
