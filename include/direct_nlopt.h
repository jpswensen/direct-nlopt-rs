/* Auto-generated C header for direct-nlopt-rs. Do not edit manually. */

#ifndef DIRECT_NLOPT_RS_H
#define DIRECT_NLOPT_RS_H

#include <stdint.h>
#include <math.h>

/**
 * Feasibility flag: point is feasible.
 */
#define FEASIBLE 0.0

/**
 * Feasibility flag: point was replaced by nearby feasible value.
 */
#define REPLACED 1.0

/**
 * Feasibility flag: point is infeasible (NaN/Inf returned from objective).
 */
#define INFEASIBLE 2.0

/**
 * Default MAXDIV matching NLOPT's DIRect.c line 60.
 */
#define PotentiallyOptimal_DEFAULT_MAX_DIV 5000



/**
 * Default relative tolerance for fglobal (0.0 means exact match required).
 */
#define DIRECT_UNKNOWN_FGLOBAL_RELTOL 0.0

/**
 * C-compatible algorithm enum, matching NLOPT's `direct_algorithm`.
 */
typedef enum direct_nlopt_algorithm {
  /**
   * Jones' original DIRECT (1993)
   */
  DIRECT_NLOPT_ALGORITHM_ORIGINAL = 0,
  /**
   * Gablonsky's locally-biased DIRECT-L (2001)
   */
  DIRECT_NLOPT_ALGORITHM_GABLONSKY = 1,
} direct_nlopt_algorithm;

/**
 * C-compatible objective function pointer, matching NLOPT's `direct_objective_func`.
 *
 * ```c
 * typedef double (*direct_objective_func)(int n, const double *x,
 *                                         int *undefined_flag,
 *                                         void *data);
 * ```
 */
typedef double (*DirectObjectiveFuncC)(int n, const double *x, int *undefined_flag, void *data);

/**
 * C-compatible result struct for `direct_nlopt_optimize_full`.
 */
typedef struct direct_nlopt_result {
  /**
   * Return code (matches NLOPT's `direct_return_code` integer values)
   */
  int return_code;
  /**
   * Number of function evaluations performed
   */
  int nfev;
  /**
   * Number of iterations performed
   */
  int nit;
} direct_nlopt_result;

/**
 * Perform global minimization using the Rust DIRECT algorithm implementation.
 *
 * This function has a C-compatible signature matching NLOPT's `direct_optimize()`,
 * enabling drop-in replacement from C/C++ code.
 *
 * # Safety
 *
 * - `f` must be a valid function pointer.
 * - `lower_bounds` and `upper_bounds` must point to arrays of length `dimension`.
 * - `x` must point to a writable array of length `dimension`.
 * - `minf` must point to a writable `double`.
 * - `force_stop` may be NULL; if non-NULL, must point to a valid `int`.
 * - `f_data` is passed through to `f` and must remain valid for the call duration.
 *
 * # Returns
 *
 * A `direct_return_code` integer matching NLOPT's convention:
 * - Positive values indicate successful termination
 * - Negative values indicate errors
 */
int direct_nlopt_optimize(DirectObjectiveFuncC f,
                          void *f_data,
                          int dimension,
                          const double *lower_bounds,
                          const double *upper_bounds,
                          double *x,
                          double *minf,
                          int max_feval,
                          int max_iter,
                          double _start,
                          double _maxtime,
                          double magic_eps,
                          double magic_eps_abs,
                          double volume_reltol,
                          double sigma_reltol,
                          const int *force_stop,
                          double fglobal,
                          double fglobal_reltol,
                          void *_logfile,
                          enum direct_nlopt_algorithm algorithm);

/**
 * Extended optimization function that also returns nfev and nit.
 *
 * Same as `direct_nlopt_optimize` but returns a `DirectResultC` struct
 * with additional statistics.
 *
 * # Safety
 *
 * Same safety requirements as `direct_nlopt_optimize`.
 */
struct direct_nlopt_result direct_nlopt_optimize_full(DirectObjectiveFuncC f,
                                                      void *f_data,
                                                      int dimension,
                                                      const double *lower_bounds,
                                                      const double *upper_bounds,
                                                      double *x,
                                                      double *minf,
                                                      int max_feval,
                                                      int max_iter,
                                                      double magic_eps,
                                                      double magic_eps_abs,
                                                      double volume_reltol,
                                                      double sigma_reltol,
                                                      const int *force_stop,
                                                      double fglobal,
                                                      double fglobal_reltol,
                                                      enum direct_nlopt_algorithm algorithm);

/**
 * Get the version string of the direct-nlopt-rs library.
 *
 * Returns a pointer to a null-terminated static string.
 * The caller must NOT free the returned pointer.
 */
const char *direct_nlopt_version(void);

#endif  /* DIRECT_NLOPT_RS_H */
