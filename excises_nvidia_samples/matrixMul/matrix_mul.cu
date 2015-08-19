#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <julia.h>
#include <cuda_runtime.h>




__global__ void matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = wA * 
    



}


void constantInit(float *data, int size, float val)
{
    for (int i=0;i<size;i++)
    {
        data[i] = val;
    }
}

int matrixMultply(dim3 &dimsA, dim3 &dimsB)
{
    float *h_A,*h_B,*h_C;
    int m, n, p;
    m = dimsA.x;
    n = dimsA.y;
    p = dimsB.y;
    size_t size_A = m*n*sizeof(float);
    size_t size_B = n*p*sizeof(float);
    size_t size_B = m*p*sizeof(float);

    cudaMallocHost((void **)h_A, size_A);
    cudaMallocHost((void **)h_B, size_B);
    cudaMallocHost((void **)h_C, size_C);
    cudaMalloc((void **)d_A,size_A);
    cudaMalloc((void **)d_B,size_B);
    cudaMalloc((void **)d_C,size_C);

    constantInit(h_A,m*n,1);
    constantInit(h_B,n*p,1);

    cudaMemcpy(d_A,h_A,size_A,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size_B,cudaMemcpyHostToDevice);
    
    dim3 dim_block(64,64);
    dim3 dim_grid(ceil(m/64),ceil(m/64)); // ceil(m/64)
    matrixMulCUDA<<<dim_block,dim_grid>>>(d_C, d_A, d_B, m, n, p);
    cudaMemcpy(d_C,h_C,size_C,cudaMemcpyDeviceToHost);
   
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    cudaError_t err = cudaGetLastError();
    
    if (cudaSuccess != err)
    {
        return -1;
    }
    return 0;
}

int main(void)
{
    int block_size = 32;

    dim3 dimsA(320, 320, 1);
    dim3 dimsB(640, 320, 1);

    int ret = matrixMultiply(dimsA, dimsB)

    exit(ret);
}
