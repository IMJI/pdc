#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include <time.h>
#include <sys/time.h>

#define N 1024

MPI_Status status;

double matrix_a[N][N],
    matrix_b[N][N],
    matrix_c[N][N];

int main(int argc, char **argv) {
    double start, end;
    int processCount, processId, slaveTaskCount, source, dest, rows, offset;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    slaveTaskCount = processCount - 1;

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    if (processId == 0) {
        srand(time(NULL));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix_a[i][j]= rand() % 10;
                matrix_b[i][j]= rand() % 10;
            }
        }

        rows = N / slaveTaskCount;
        offset = 0;

        for (dest=1; dest <= slaveTaskCount; dest++) {
            MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&matrix_a[offset][0], rows * N, MPI_DOUBLE,dest,1, MPI_COMM_WORLD);
            MPI_Send(&matrix_b, N * N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
            
            offset = offset + rows;
        }

        for (int i = 1; i <= slaveTaskCount; i++) {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&matrix_c[offset][0], rows * N, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
        }
    }

    if (processId > 0) {
        source = 0;

        MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&matrix_a, rows * N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&matrix_b, N * N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);

        for (int k = 0; k < N; k++) {
            for (int i = 0; i < rows; i++) {
                matrix_c[i][k] = 0.0;
                for (int j = 0; j < N; j++)
                matrix_c[i][k] = matrix_c[i][k] + matrix_a[i][j] * matrix_b[j][k];
            }
        }

        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&matrix_c, rows * N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    if (processId == 0) {
        printf("Time = %f\n", end - start);
    }

    MPI_Finalize();
}