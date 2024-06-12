#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>

#define N 1024

double matrix_a[N][N],
    matrix_b[N][N],
    matrix_c[N][N];

int main(int argc, char *argv[]) {
    int i, j, k;
    double start, end;
    int threads = atoi(argv[1]);

    start = omp_get_wtime();
    
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix_a[i][j]= rand() % 10;
            matrix_b[i][j]= rand() % 10;
        }
    }

    omp_set_dynamic(0);
    
    #pragma omp parallel for shared(matrix_a, matrix_b, matrix_c) private(i, j, k) num_threads(threads)
    for(i = 0; i < N; i++){
        for(j = 0; j < N; j++){
            matrix_c[i][j] = 0.0;
            for(k = 0; k < N; k++)
                matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j];
        }
    }

    end = omp_get_wtime();
    printf("Time = %lf\n", end - start);
} 