#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>


void run_test(int argc, char **argv);

extern "C"
void compute_cpu(float *reference, float *idata, const unsigned int len);

__global__ void test_kernel(float *g_idata, float *g_odata)
{
    extern __shared__ float sdata[];

    const unsigned int tid = threadIdx.x;
    const unsigned int num_threads_per_block = blockDim.x;

    sdata[tid] = g_idata[tid];
    __syncthreads();

    sdata[tid] = (float) num_threads_per_block * sdata[tid];
    __syncthreads();

    g_odata[tid] = sdata[tid];
}
int main(int argc, char **argv)
{
    run_test(argc, argv);
}

void run_test(int argc, char **argv)
{
    bool bTestResult = true;

    unsigned int num_threads_per_block = 32;
    unsigned int mem_size = sizeof(float) * num_threads_per_block;

    float *h_idata = (float *)malloc(mem_size);
    
    for (unsigned int i=0; i < num_threads_per_block;i++ )
    {
        h_idata[i] = (float)i;
    }

    float *d_idata;
    cudaMalloc((void **)&d_idata, mem_size);
    cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice);

    float *d_odata;
    cudaMalloc((void **)&d_odata, mem_size);

    dim3 grid(1,1,1);
    dim3 threads(num_threads_per_block,1,1);

    test_kernel<<<grid,threads, mem_size>>>(d_idata, d_odata);

    float *h_odata = (float *)malloc(mem_size);
    cudaMemcpy(h_odata,d_odata,sizeof(float)*num_threads, cudaMemcpyDeviceToHost);


}

