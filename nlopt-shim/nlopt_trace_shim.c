/* nlopt_trace_shim.c — C tracing wrapper for NLOPT DIRECT algorithm.
 *
 * Replicates the main loop from DIRect.c's direct_direct_() with matching
 * trace printf statements at the same algorithm points as the Rust
 * implementation's trace feature.
 *
 * Output format matches the Rust TRACE lines for line-by-line comparison. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "direct-internal.h"

/* Run NLOPT DIRECT with tracing and write trace output to the provided buffer.
 * This function replicates direct_direct_() from DIRect.c with added trace
 * output at key algorithm points.
 *
 * Parameters match direct_optimize() wrapper:
 *   fcn, n, eps, epsabs, maxf, maxt, minf, l, u, algmethod, fglobal, fglper,
 *   volper, sigmaper
 *
 * trace_buf: output buffer for trace lines
 * trace_buf_size: size of trace_buf
 *
 * Returns: NLOPT's ierror code */
int nlopt_trace_direct(
    direct_objective_func fcn, void *fcn_data,
    int n, double eps_in, double epsabs,
    int maxf, int maxt_in,
    double *minf_out, double *x_out,
    const double *lower, const double *upper,
    int algmethod,
    double fglobal, double fglper_in,
    double volper_in, double sigmaper_in,
    char *trace_buf, int trace_buf_size)
{
    /* Internal variables matching DIRect.c */
    int MAXFUNC, MAXDEEP, MAXDIV;
    int ierror = 0;
    int jones = algmethod;
    double eps = eps_in;
    double epsfix;
    int iepschange;
    int oops;
    int t, tstart;
    int actdeep, actmaxdeep, actdeep_div__;
    int maxi, maxpos;
    int numfunc;
    int ifeasiblef, iinfesiblef;
    double fmax;
    int minpos;
    double delta;
    int oldpos;
    int newtosample;
    int start;
    int help;
    double divfactor;
    int oldmaxf;
    int increase;
    int cheat = 0;
    double kmax = 1e10;

    /* Working arrays */
    double *c__, *f, *thirds, *levels, *w, *x, *l, *u;
    int *length, *point, *anchor, *arrayi, *list2, *s;
    int ifree;

    int buf_pos = 0;

#define TRACE_PRINTF(...) do { \
    if (buf_pos < trace_buf_size - 1) { \
        buf_pos += snprintf(trace_buf + buf_pos, trace_buf_size - buf_pos, __VA_ARGS__); \
    } \
} while(0)

    /* Compute sizes (matching DIRect.c lines 342-355) */
    int n_val = n;
    int maxf_val = maxf;
    int maxt_val = maxt_in;

    /* Match dirheader_ size computation */
    MAXFUNC = (maxf_val > 0) ? maxf_val + 20 : 5020;
    MAXDEEP = (int)(log((double)MAXFUNC) / log(2.0)) + 10;
    if (MAXDEEP < 600) MAXDEEP = 600;
    MAXDIV = 5000;

    /* Allocate arrays */
    c__ = (double*)calloc((size_t)MAXFUNC * n, sizeof(double));
    f = (double*)calloc((size_t)MAXFUNC * 2, sizeof(double));
    thirds = (double*)calloc(MAXDEEP + 1, sizeof(double));
    levels = (double*)calloc(MAXDEEP + 1, sizeof(double));
    w = (double*)calloc(n, sizeof(double));
    x = (double*)calloc(n + 1, sizeof(double));
    l = (double*)calloc(n + 1, sizeof(double));
    u = (double*)calloc(n + 1, sizeof(double));
    length = (int*)calloc((size_t)MAXFUNC * n, sizeof(int));
    point = (int*)calloc(MAXFUNC, sizeof(int));
    anchor = (int*)calloc(MAXDEEP + 2, sizeof(int));
    arrayi = (int*)calloc(n, sizeof(int));
    list2 = (int*)calloc(n, sizeof(int));
    s = (int*)calloc(MAXDIV * 2, sizeof(int));

    /* Copy bounds to 1-based arrays (matching DIRect.c convention) */
    for (int i = 0; i < n; i++) {
        l[i + 1] = lower[i];
        u[i + 1] = upper[i];
    }

    /* Preprocessing */
    direct_dirpreprc_(&u[1], &l[1], &n_val, &l[1], &u[1], &oops);
    if (oops > 0) { ierror = -3; goto cleanup; }

    /* Epsilon handling (matching dirheader_) */
    if (eps < 0.0) {
        iepschange = 1;
        epsfix = -eps;
        eps = -eps;
    } else {
        iepschange = 0;
        epsfix = 1e100;
    }

    /* divfactor */
    if (fglobal == 0.0) divfactor = 1.0;
    else divfactor = fabs(fglobal);

    oldmaxf = maxf_val;
    increase = 0;

    /* Initialize lists */
    direct_dirinitlist_(anchor, &ifree, point, f, &MAXFUNC, &MAXDEEP);

    /* Initialize algorithm */
    double minf_local = 0.0;
    int force_stop = 0;
    direct_dirinit_(f, fcn, c__, length, &actdeep, point, anchor, &ifree,
        NULL, arrayi, &maxi, list2, w, &x[1], &l[1], &u[1],
        &minf_local, &minpos, thirds, levels, &MAXFUNC, &MAXDEEP, &n_val, &n_val,
        &fmax, &ifeasiblef, &iinfesiblef, &ierror, fcn_data, jones,
        0.0, 0.0, &force_stop);

    if (ierror < 0) goto cleanup;

    numfunc = maxi + 1 + maxi;
    actmaxdeep = 1;
    tstart = 2;

    /* Trace init: log sample points from initialization */
    /* After dirinit_, the linked list is rearranged by dirdivide_/dirinsertlist_.
       Use direct rect indices instead: for init with n dims, pos_idx = 2+2*k, neg_idx = 3+2*k */
    {
        int init_maxi = n;
        for (int k = 0; k < init_maxi; k++) {
            int dim_1based = k + 1;
            int pos_idx = 2 + 2 * k;
            int neg_idx = 3 + 2 * k;
            double pos_f = f[(pos_idx << 1) - 2];
            double neg_f = f[(neg_idx << 1) - 2];
            TRACE_PRINTF("TRACE SAMPLE rect=1 dim=%d pos_idx=%d pos_f=%.17e neg_idx=%d neg_f=%.17e\n",
                dim_1based, pos_idx, pos_f, neg_idx, neg_f);
        }
    }

    /* Trace: initialization result */
    {
        double center_f = f[0]; /* f[1,1] = f[(1<<1)-2] = f[0] */
        TRACE_PRINTF("TRACE INIT center_f=%.17e minf=%.17e minpos=%d nfev=%d\n",
            center_f, minf_local, minpos, numfunc);
    }

    /* Main loop */
    for (t = tstart; t <= maxt_val; ++t) {
        actdeep = actmaxdeep;
        direct_dirchoose_(anchor, s, &MAXDEEP, f, &minf_local, eps, epsabs, levels, &maxpos,
            length, &MAXFUNC, &MAXDEEP, &MAXDIV, &n_val, NULL, &cheat, &kmax,
            &ifeasiblef, jones);

        if (algmethod == 0) {
            direct_dirdoubleinsert_(anchor, s, &maxpos, point, f, &MAXDEEP, &MAXFUNC,
                &MAXDIV, &ierror);
            if (ierror == -6) goto cleanup;
        }

        /* Trace: log iteration start with selected rects */
        {
            int selected_count = 0;
            for (int j = 1; j <= maxpos; j++) {
                if (s[j - 1] > 0) selected_count++;
            }
            TRACE_PRINTF("TRACE ITER t=%d selected=%d minf=%.17e nfev=%d\n",
                t, selected_count, minf_local, numfunc);
            int sel_idx = 0;
            for (int j = 1; j <= maxpos; j++) {
                if (s[j - 1] > 0) {
                    int rect_idx = s[j - 1];
                    int rect_level = s[j + MAXDIV - 1];
                    double rect_f = f[(rect_idx << 1) - 2];
                    TRACE_PRINTF("TRACE SELECT j=%d rect=%d level=%d f=%.17e\n",
                        sel_idx, rect_idx, rect_level, rect_f);
                    sel_idx++;
                }
            }
        }

        oldpos = minpos;
        newtosample = 0;

        for (int j = 1; j <= maxpos; ++j) {
            actdeep = s[j + MAXDIV - 1];
            if (s[j - 1] > 0) {
                actdeep_div__ = direct_dirgetmaxdeep_(&s[j - 1], length, &MAXFUNC, &n_val);
                delta = thirds[actdeep_div__ + 1];
                actdeep = s[j + MAXDIV - 1];

                if (actdeep + 1 >= MAXDEEP) {
                    ierror = -6;
                    goto done;
                }

                if (actdeep > actmaxdeep) actmaxdeep = actdeep;
                help = s[j - 1];

                /* Remove from anchor list */
                if (!(anchor[actdeep + 1] == help)) {
                    int pos1 = anchor[actdeep + 1];
                    while (!(point[pos1 - 1] == help)) {
                        pos1 = point[pos1 - 1];
                    }
                    point[pos1 - 1] = point[help - 1];
                } else {
                    anchor[actdeep + 1] = point[help - 1];
                }

                if (actdeep < 0) {
                    actdeep = (int)f[(help << 1) - 2];
                }

                direct_dirget_i__(length, &help, arrayi, &maxi, &n_val, &MAXFUNC);

                direct_dirsamplepoints_(c__, arrayi, &delta, &help, &start, length,
                    NULL, f, &ifree, &maxi, point, &x[1], &l[1], &minf_local, &minpos,
                    &u[1], &n_val, &MAXFUNC, &MAXDEEP, &oops);
                if (oops > 0) { ierror = -4; goto cleanup; }

                newtosample += maxi;

                direct_dirsamplef_(c__, arrayi, &delta, &help, &start, length,
                    NULL, f, &ifree, &maxi, point, fcn, &x[1], &l[1], &minf_local,
                    &minpos, &u[1], &n_val, &MAXFUNC, &MAXDEEP, &oops, &fmax,
                    &ifeasiblef, &iinfesiblef, fcn_data, &force_stop);
                if (oops > 0) { ierror = -5; goto cleanup; }

                /* Trace: log sample points and f-values */
                {
                    int pos = start;
                    for (int k = 0; k < maxi; k++) {
                        int dim_1based = arrayi[k];
                        int pos_idx = pos;
                        double pos_f = f[(pos_idx << 1) - 2];
                        int neg_idx = point[pos_idx - 1];
                        double neg_f = f[(neg_idx << 1) - 2];
                        TRACE_PRINTF("TRACE SAMPLE rect=%d dim=%d pos_idx=%d pos_f=%.17e neg_idx=%d neg_f=%.17e\n",
                            help, dim_1based, pos_idx, pos_f, neg_idx, neg_f);
                        pos = point[neg_idx - 1];
                    }
                }

                direct_dirdivide_(&start, &actdeep_div__, length, point, arrayi,
                    &help, list2, w, &maxi, f, &MAXFUNC, &MAXDEEP, &n_val);

                direct_dirinsertlist_(&start, anchor, point, f, &maxi, length,
                    &MAXFUNC, &MAXDEEP, &n_val, &help, jones);

                /* Trace: log division result */
                {
                    char lengths_str[1024] = {0};
                    int lpos = 0;
                    for (int d = 0; d < n; d++) {
                        if (d > 0) lpos += snprintf(lengths_str + lpos, sizeof(lengths_str) - lpos, ",");
                        /* length stored as length[(rect-1)*n + dim] (row-major, stride n) */
                        lpos += snprintf(lengths_str + lpos, sizeof(lengths_str) - lpos, "%d",
                            length[(help - 1) * n + d]);
                    }
                    char dims_str[1024] = {0};
                    int dpos = 0;
                    for (int k = 0; k < maxi; k++) {
                        if (k > 0) dpos += snprintf(dims_str + dpos, sizeof(dims_str) - dpos, ",");
                        dpos += snprintf(dims_str + dpos, sizeof(dims_str) - dpos, "%d", arrayi[k]);
                    }
                    TRACE_PRINTF("TRACE DIVIDE rect=%d maxi=%d dims=[%s] lengths_after=[%s]\n",
                        help, maxi, dims_str, lengths_str);
                }

                numfunc = numfunc + maxi + maxi;
            }
        }

        /* Trace: end of iteration */
        TRACE_PRINTF("TRACE ENDITER t=%d minf=%.17e minpos=%d nfev=%d\n",
            t, minf_local, minpos, numfunc);

        /* Termination checks */
        ierror = jones;
        jones = 0;
        actdeep_div__ = direct_dirgetlevel_(&minpos, length, &MAXFUNC, &n_val, jones);
        jones = ierror;

        {
            double vol_delta = thirds[actdeep_div__] * 100.0;
            if (volper_in > 0.0 && vol_delta <= volper_in) {
                ierror = 4; /* VOLTOL */
                goto done;
            }
        }

        {
            ierror = jones;
            jones = 0;
            int sigma_level = direct_dirgetlevel_(&minpos, length, &MAXFUNC, &n_val, jones);
            jones = ierror;
            /* Actually need jones-based level for sigma */
            sigma_level = direct_dirgetlevel_(&minpos, length, &MAXFUNC, &n_val, ierror);
            double sigma_delta = levels[sigma_level];
            if (sigmaper_in > 0.0 && sigma_delta <= sigmaper_in) {
                ierror = 5; /* SIGMATOL */
                goto done;
            }
        }

        if (fglobal > -1e99 && fglper_in > 0.0) {
            if ((minf_local - fglobal) * 100.0 / divfactor <= fglper_in) {
                ierror = 3; /* GLOBAL_FOUND */
                goto done;
            }
        }

        /* Replace infeasible */
        if (iinfesiblef > 0) {
            direct_dirreplaceinf_(&ifree, &ifeasiblef, f, c__, thirds,
                length, anchor, point, &x[1], &l[1], &u[1], &MAXFUNC,
                &MAXDEEP, &n_val, &n_val, &fmax, jones);
        }

        /* Epsilon update */
        if (iepschange == 1) {
            eps = (fabs(minf_local) > epsfix) ? epsfix / fabs(minf_local) : epsfix;
        }

        /* Infeasible budget extension */
        if (increase) {
            if (maxf_val > 0) maxf_val = numfunc + oldmaxf;
            if (ifeasiblef == 0) increase = 0;
        }

        /* Max feval check */
        if (maxf_val > 0 && numfunc > maxf_val) {
            if (ifeasiblef == 0) {
                ierror = 1; /* MAXFEVAL_EXCEEDED */
                goto done;
            } else {
                increase = 1;
                if (oldmaxf > 0) maxf_val = numfunc + oldmaxf;
            }
        }
    }

    /* Loop finished = maxiter exceeded */
    ierror = 2;

done:
    /* Extract best point — matching DIRect.c line 734:
       x[i__] = c__[i__ + minpos * i__1 - i__1 - 1] * l[i__] + l[i__] * u[i__]
       where i__1 = n, l[i__] = xs1, u[i__] = xs2 after dirpreprc_ */
    *minf_out = minf_local;
    for (int i = 0; i < n; i++) {
        x_out[i] = c__[(minpos - 1) * n + i] * l[i + 1] + l[i + 1] * u[i + 1];
    }

cleanup:
    free(c__); free(f); free(thirds); free(levels);
    free(w); free(x); free(l); free(u);
    free(length); free(point); free(anchor);
    free(arrayi); free(list2); free(s);

    return ierror;
}
