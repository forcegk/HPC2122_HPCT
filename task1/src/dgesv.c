#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if defined(__INTEL_COMPILER)
#include "mkl_lapacke.h"
#elif defined(__GNUC__) || defined(__GNUG__)
#include <math.h>
#include "lapacke.h"
#endif

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

int my_dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb) {
    int i, j, k;
    double *l, *u, *z;

    l = (double *) calloc(n * n, sizeof(double));
    u = (double *) calloc(n * n, sizeof(double));
    z = (double *) calloc(n * n, sizeof(double));

    // compute the LU decomposition of a
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (j < i) {
                l[j * n + i] = 0;
            } else {
                l[j * n + i] = a[j * n + i];
                for (k = 0; k < i; k++) {
                    l[j * n + i] -= l[j * n + k] * u[k * n + i];
                }
            }
        }
        for (j = 0; j < n; j++) {
            if (j < i) {
                u[i * n + j] = 0;
            } else if (j == i) {
                u[i * n + j] = 1;
            } else {
                u[i * n + j] = a[i * n + j] / l[i * n + i];
                for (k = 0; k < i; k++) {
                    u[i * n + j] -= (l[i * n + k] * u[k * n + j])
                                   / l[i * n + i];
                }
            }
        }
    }

    // forward substitution
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            z[j * n + i] = b[j * n + i];
            for (k = 0; k < j; k++) {
                z[j * n + i] -= l[j * n + k] * z[k * n + i];
            }
            z[j * n + i] /= l[j * n + j];
        }
    }

    // backward substitution
    for (i = 0; i < n; i++) {
        for (j = n - 1; j >= 0; j--) {
            b[j * n + i] = z[j * n + i];
            for (k = n - 1; k > j; k--) {
                b[j * n + i] -= u[j * n + k] * b[k * n + i];
            }
            b[j * n + i] /= u[j * n + j];
        }
    }

    free(l);
    free(u);
    free(z);

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
