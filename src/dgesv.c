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

    aligned_double *restrict l, *restrict u, *restrict z, *restrict x;

    aligned_double *restrict A, *restrict B;
    aligned_double *restrict L, *restrict U, *restrict UU, *restrict Z, *restrict X;

    l = (aligned_double *) calloc(n * n, sizeof(aligned_double));
    u = (aligned_double *) calloc(n * n, sizeof(aligned_double));
    z = (aligned_double *) calloc(n * n, sizeof(aligned_double));
    x = (aligned_double *) calloc(n * n, sizeof(aligned_double));

#if defined(_OPENMP)
    aligned_double temp_double; 
#endif

    // compute the LU decomposition of a
    for (i = 0; i < n; i++) {
        L = l + i * n;
        A = a + i * n;

        for (j = i; j < n; j++) {
#if defined(_OPENMP)
            temp_double = A[i];
#else
            L[i] = A[i];
            U = u;
#endif

#pragma code_align 32
#pragma omp parallel for private(U) reduction(-:temp_double) if(i > 256)
            for (k = 0; k < i; k++) {
#if defined(_OPENMP)
                    U = u + k*n;
                    temp_double -= L[k] * U[i];
#else
                    L[i] -= L[k] * U[i];
                    U += n;
#endif
            }
#if defined(_OPENMP)
            L[i] = temp_double;
#endif
            L += n;
            A += n;
        }

        L = l + i * n;
        A = a + i * n;
        U = u + i * n;
        U[i] = 1;
        for (j = i + 1; j < n; j++) {


#if defined(_OPENMP)
            temp_double = A[j];
#else
            U[j] = A[j];
            UU = u;
#endif

#pragma code_align 32
#pragma omp parallel for private(UU) reduction(-:temp_double) if(i > 256)
            for (k = 0; k < i; k++) {
#if defined(_OPENMP)
                UU = u + k*n;
                temp_double -= L[k] * UU[j];
#else
                U[j] -= L[k] * UU[j];
                UU += n;
#endif
            }
#if defined(_OPENMP)
            U[j] = temp_double / L[i];
#else
            U[j] /= L[i];
#endif
        }
    }

    // forward substitution
    Z = z;
    for (i = 0; i < n; i++) {
        B = b;
        L = l;
        for (j = 0; j < n; j++) {
            Z[j] = B[i];
            for (k = 0; k < j; k++) {
                Z[j] -= L[k] * Z[k];
            }
            Z[j] /= L[j];
            B += n;
            L += n;
        }
        Z += n;
    }

    // backward substitution
    X = x;
    Z = z;
    for (i = 0; i < n; i++) {
        U = u + n * n - n;
        for (j = n - 1; j >= 0; j--) {
            X[j] = Z[j];
            for (k = n - 1; k > j; k--) {
                X[j] -= U[k] * X[k];
            }
            X[j] /= U[j];
            U -= n;
        }
        X += n;
        Z += n;
    }

    // transpose x
#pragma omp parallel for private(i,j)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            b[i * n + j] = x[j * n + i];
        }
    }

    free(l);
    free(u);
    free(z);
    free(x);

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
