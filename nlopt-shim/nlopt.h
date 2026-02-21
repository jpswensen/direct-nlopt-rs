/* Minimal nlopt.h shim for compiling DIRECT algorithm files standalone.
   Only provides the typedefs needed by nlopt-util.h */
#ifndef NLOPT_H
#define NLOPT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef double (*nlopt_func)(unsigned n, const double *x,
                             double *gradient, void *func_data);
typedef void (*nlopt_mfunc)(unsigned m, double *result,
                            unsigned n, const double *x,
                            double *gradient, void *func_data);
typedef void (*nlopt_precond)(unsigned n, const double *x,
                              const double *v, double *vpre, void *data);

typedef enum {
    NLOPT_FAILURE = -1,
    NLOPT_INVALID_ARGS = -2,
    NLOPT_OUT_OF_MEMORY = -3,
    NLOPT_ROUNDOFF_LIMITED = -4,
    NLOPT_FORCED_STOP = -5,
    NLOPT_SUCCESS = 1,
    NLOPT_STOPVAL_REACHED = 2,
    NLOPT_FTOL_REACHED = 3,
    NLOPT_XTOL_REACHED = 4,
    NLOPT_MAXEVAL_REACHED = 5,
    NLOPT_MAXTIME_REACHED = 6
} nlopt_result;

typedef struct nlopt_opt_s *nlopt_opt;

typedef enum {
    NLOPT_GN_DIRECT = 0,
    NLOPT_GN_DIRECT_L
} nlopt_algorithm;

#define NLOPT_EXTERN(T) extern T

#ifdef __cplusplus
}
#endif

#endif /* NLOPT_H */
