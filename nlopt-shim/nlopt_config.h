/* Minimal nlopt_config.h for compiling DIRECT algorithm files standalone */
#ifndef NLOPT_CONFIG_H
#define NLOPT_CONFIG_H

#define HAVE_COPYSIGN 1
#define HAVE_GETTIMEOFDAY 1
#define HAVE_TIME 1
#define HAVE_SYS_TIME_H 1
#define HAVE_UNISTD_H 1
#define HAVE_STDINT_H 1
#define HAVE_ISNAN 1
#define HAVE_ISINF 1
#define HAVE_FPCLASSIFY 1

#define THREADLOCAL _Thread_local

#define SIZEOF_UNSIGNED_INT 4
#define SIZEOF_UNSIGNED_LONG 8

#define MAJOR_VERSION 2
#define MINOR_VERSION 9
#define BUGFIX_VERSION 0

#endif /* NLOPT_CONFIG_H */
