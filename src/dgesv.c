#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <errno.h>

#if defined(__INTEL_COMPILER)
#include "mkl_lapacke.h"
#elif defined(__GNUC__) || defined(__GNUG__)
#include <math.h>
#include "lapacke.h"
#endif

typedef double aligned_double __attribute__((aligned (32)));

double *generate_matrix(int size) {
    int i;
    double *matrix = (double *) malloc(sizeof(double) * size * size);

    srand(1);

    for (i = 0; i < size * size; i++) {
        matrix[i] = rand() % 100;
    }

    return matrix;
}

int is_nearly_equal(double x, double y) {
    const double epsilon = 1e-5;
    return abs(x - y) <= epsilon * abs(x);
}

int check_result(double *bref, double *b, int size) {
    int i;

    for (i = 0; i < size * size; i++) {
        if (!is_nearly_equal(bref[i], b[i]))
            return 0;
    }

    return 1;
}

int my_dgesv(int n, __attribute__((unused)) int nrhs, double *restrict a, __attribute__((unused)) int lda, __attribute__((unused)) int *ipiv, double *restrict b, __attribute__((unused)) int ldb) {
    int i, j, k;

    aligned_double *restrict l, *restrict u;
    aligned_double *restrict z, *restrict x, *restrict y;

    l = (aligned_double *) calloc(n * n, sizeof(aligned_double));
    u = (aligned_double *) calloc(n * n, sizeof(aligned_double));
    z = (aligned_double *) calloc(n * n, sizeof(aligned_double));
    x = (aligned_double *) calloc(n * n, sizeof(aligned_double));
    y = (aligned_double *) calloc(n * n, sizeof(aligned_double));

#pragma omp parallel private(i)
{
    // compute the LU decomposition of a
    for (i = 0; i < n; i++) {
#pragma omp for private(k) schedule(static)
        for (j = i; j < n; j++) {
            l[j * n + i] = a[j * n + i];
#pragma ivdep
            for (k = 0; k < i; k++) {
                l[j * n + i] -= l[j * n + k] * y[i * n + k];
            }
        }

#pragma omp master
        y[i * n + i] = 1;

#pragma omp for private(k) schedule(static)
        for (j = i + 1; j < n; j++) {
            y[j * n + i] = a[i * n + j];
#pragma ivdep
            for (k = 0; k < i; k++) {
                y[j * n + i] -= l[i * n + k] * y[j * n + k];
            }
            y[j * n + i] /= l[i * n + i];
        }
    }

    // transpose u
#pragma omp for private(j) schedule(static)
    for (i = 0; i < n; i++) {
#pragma ivdep
        for (j = 0; j < n; j++) {
            u[i * n + j] = y[j * n + i];
        }
    }

    // forward substitution
#pragma omp for private(j,k) schedule(static) nowait
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            z[i * n + j] = b[j * n + i];
#pragma ivdep
            for (k = 0; k < j; k++) {
                z[i * n + j] -= l[j * n + k] * z[i * n + k];
            }
            z[i * n + j] /= l[j * n + j];
        }
    }

    // backward substitution
#pragma omp for private(j,k) schedule(static)
    for (i = 0; i < n; i++) {
        for (j = n - 1; j >= 0; j--) {
            x[i * n + j] = z[i * n + j];
#pragma ivdep
            for (k = n - 1; k > j; k--) {
                x[i * n + j] -= u[j * n + k] * x[i * n + k];
            }
            x[i * n + j] /= u[j * n + j];
        }
    }

    // transpose x
#pragma omp for private(j) schedule(static)
    for (i = 0; i < n; i++) {
#pragma ivdep
        for (j = 0; j < n; j++) {
            b[i * n + j] = x[j * n + i];
        }
    }
} // #pragma omp parallel private(i)

    free(l);
    free(u);
    free(z);
    free(x);
    free(y);

    return 0;
}

int main(int argc, char *argv[]) {
    int size, status;

    double *a, *aref;
    double *b, *bref;

    // very basic error check, but DAMN I am tired of segfaults when I forget the param hehe
    if(argc > 1){
        size = strtoul(argv[1], NULL, 0);
    } else {
        errno = EFAULT;
    }

    if(size<=0 || errno){
        fprintf(stderr, "Syntax: \"%s n\", where:\n\t- n is a positive integer number\n\n", argv[0]);
        exit(1);
    }

    a = generate_matrix(size);
    b = generate_matrix(size);
    aref = generate_matrix(size);
    bref = generate_matrix(size);

#if defined(__INTEL_COMPILER)
    MKL_INT n = size, nrhs = size, lda = size, ldb = size, info;
    MKL_INT *ipiv = (MKL_INT *)malloc(sizeof(MKL_INT) * size);
#elif defined(__GNUC__) || defined(__GNUG__)
    int n = size, nrhs = size, lda = size, ldb = size, info;
    int *ipiv = (int *)malloc(sizeof(int) * size);
#endif

    struct timeval tStart, tEnd;
    gettimeofday(&tStart, NULL);
    info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, aref, lda, ipiv, bref, ldb);
    gettimeofday(&tEnd, NULL);

    double time = (double) (tEnd.tv_usec - tStart.tv_usec) / 1000000
                  + (double) (tEnd.tv_sec - tStart.tv_sec);

#if defined(__INTEL_COMPILER)
    printf("Time taken by MKL: %.2fs\n", time);
    MKL_INT *ipiv2 = (MKL_INT *)malloc(sizeof(MKL_INT) * size);
#elif defined(__GNUC__) || defined(__GNUG__)
    printf("Time taken by LAPACK: %.2fs\n", time);
    int *ipiv2 = (int *)malloc(sizeof(int) * size);
#endif

    gettimeofday(&tStart, NULL);
    my_dgesv(n, nrhs, a, lda, ipiv2, b, ldb);
    gettimeofday(&tEnd, NULL);

    time = (double) (tEnd.tv_usec - tStart.tv_usec) / 1000000
                  + (double) (tEnd.tv_sec - tStart.tv_sec);

    printf("Time taken by my implementation: %.2fs\n", time);

    if (check_result(bref, b, size) == 1) {
        printf("Result is ok!\n");
        status = 0;
    } else {
        printf("Result is wrong!\n");
        status = 1;
    }

    free(ipiv);
    free(ipiv2);
    free(a);
    free(b);
    free(aref);
    free(bref);

    return status;
}
