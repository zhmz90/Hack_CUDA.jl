#include <stdio.h>
#include <stdlib.h>
#include <julia.h>
#include <cuda_runtime.h>

__global__ void vec_add(float *d_A,float *d_B,float *d_C, int len)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx<len )
    {
        d_C[idx] = d_A[idx] + d_B[idx];
    }
}


int main(void)
{
    // init julia
    jl_init("/home/guo/julia/usr/lib");
    jl_eval_string("println(\"julia: julia init successfully\")");

    float *h_A;
    // vectAdd and length(vec) = 10000
    int block_len = 1024;
    int num_block = 1024*16;
    int len = block_len*num_block;
    size_t size = len*sizeof(float);
    // malloc host mem use cudaMallocHost
    cudaMallocHost((void **)&h_A, size);
    //init vector A,B,C
    for (int i=0;i<len;i++)
    {
        h_A[i] = rand() % len;
    }
    jl_eval_string("println(\"julia: host A,B,C init successfully\")");
   
    // alloc device mem
    float *d_A;
    cudaMalloc((void **)&d_A,size);

    //set call stack height
  //  printf("cuda call stack heigth is %d\n", (int)cudaDeviceGetLimit(cudaLimitDevRuntimeSyncDepth));
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth,10);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // memcpy ==> vec_add<<<>>> ==> memcpy #TODO stream and event async parallel maxthroughtput
    // data transfer from host to device
    cudaDeviceSynchronize();
    cudaEventRecord(start,0);
    cudaMemcpyAsync(d_A,h_A,size,cudaMemcpyHostToDevice,0);

    dim3 dimGrid(num_block,1);
    dim3 dimBlock(block_len,1);
    vec_add<<<dimGrid,dimBlock,0,0>>>(d_A,len);
//    cudaDeviceSynchronize();

    cudaMemcpyAsync(h_A,d_A,size,cudaMemcpyDeviceToHost,0); // cudaMemcpy is a synchronize function
    cudaDeviceSynchronize();
    cudaEventRecord(stop,0);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time,start,stop);
    printf("gpu_time is %f\n",gpu_time);
    // check result 
    bool ret = true;
    for (int i = 0; i< len;i++)
    {
        printf("h_A[%d] is %f\t",i,h_A[i]);
    
    }
    ret ? printf("result is passed\n") : printf("result is failed\n");

    //Free objects
    cudaFree(d_A);
    cudaFreeHost(h_A);

    jl_atexit_hook(0);

    cudaError_t err;
    err = cudaDeviceReset();
    if (err  != cudaSuccess)
    {
        printf("cudaDeviceReset failed\n");
    }

    return 0;
}
