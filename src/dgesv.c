#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if defined(__INTEL_COMPILER)
#include "mkl_lapacke.h"
#elif defined(__GNUC__) || defined(__GNUG__)
#include <math.h>
#include "lapacke.h"
#endif

typedef double aligned_double __attribute__((aligned (16)));

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

    // compute the LU decomposition of a
    for (i = 0; i < n; i++) {
        L = l + i * n;
        A = a + i * n;
        for (j = i; j < n; j++) {
            L[i] = A[i];
            U = u;
            for (k = 0; k < i; k++) {
                L[i] -= L[k] * U[i];
                U += n;
            }
            L += n;
            A += n;
        } 

        L = l + i * n;
        A = a + i * n;
        U = u + i * n;
        U[i] = 1;
        for (j = i + 1; j < n; j++) {
            U[j] = A[j];
            UU = u;
            for (k = 0; k < i; k++) {
                U[j] -= L[k] * UU[j];
                UU += n;
            }
            U[j] /= L[i];
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
    int size;

    double *a, *aref;
    double *b, *bref;

    size = atoi(argv[1]);

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

    clock_t tStart = clock();
    info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, aref, lda, ipiv, bref, ldb);

#if defined(__INTEL_COMPILER)
    printf("Time taken by MKL: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
    MKL_INT *ipiv2 = (MKL_INT *)malloc(sizeof(MKL_INT) * size);
#elif defined(__GNUC__) || defined(__GNUG__)
    printf("Time taken by LAPACK: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
    int *ipiv2 = (int *)malloc(sizeof(int) * size);
#endif

    tStart = clock();
    my_dgesv(n, nrhs, a, lda, ipiv2, b, ldb);
    printf("Time taken by my implementation: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

    if (check_result(bref, b, size) == 1) {
        printf("Result is ok!\n");
    } else {
        printf("Result is wrong!\n");
    }

    return 0;
}
