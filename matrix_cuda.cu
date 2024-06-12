#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;


#define SIZE 10000
#define M_SIZE 100000000
#define BLOCK_SIZE 10

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc((void**) &d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc((void**) &d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc((void**) &d_C.elements, size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    cudaThreadSynchronize();

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}

void gen_random_numbers(float *array, int len, int min, int max){
    for (int i = 0; i < len; i++)
        array[i] = rand() % (max - min + 1) + min;
}

int main() {
    Matrix A;
    A.width = SIZE;
    A.height = SIZE;
    float *A_numbers = (float*) malloc(sizeof(float)*M_SIZE);
    gen_random_numbers(A_numbers, M_SIZE, 10, 100);
    A.elements = A_numbers;

    Matrix B;
    B.width = SIZE;
    B.height = SIZE;
    float *B_numbers = (float*) malloc(sizeof(float)*M_SIZE);
    gen_random_numbers(B_numbers, M_SIZE, 10, 100);
    B.elements = B_numbers;

    Matrix C;
    C.width = SIZE;
    C.height = SIZE;
    float *C_numbers = (float*) malloc(sizeof(float)*M_SIZE);
    C.elements = C_numbers;

    
    clock_t start, finish;
    
    start = clock();
    MatMul(A, B, C);
    finish = clock();
    printf("Time = %f\n", ((float) (finish - start)) / CLOCKS_PER_SEC);

    return 0;
}
