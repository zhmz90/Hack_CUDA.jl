#include <stdio.h>
#include <stdlib.h>
#include <julia.h>
#include <cuda_runtime.h>

__global__ void vec_add(float *d_A,float *d_B,float *d_C, int len)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    int idx = threadIdx.x;
    if ( idx<len )
    {
        d_C[idx] = d_A[idx] + d_B[idx];

        /*
        printf("d_A[%d] is %f\n",idx,d_A[idx]);
        printf("d_B[%d] is %f\n",idx,d_B[idx]);

        printf("d_C[%d] is %f\n",idx,d_C[idx]);
        */
    }
}

int main(void)
{
    // init julia
    jl_init("/home/guo/julia/usr/lib");
    jl_eval_string("println(\"julia: julia init successfully\")");

    printf("sizeof(double) is %d\n", (int)sizeof(double));
    printf("sizeof(int) is %d\n", (int)sizeof(int));
    printf("sizeof(float) is %d\n", (int)sizeof(float));

    float *h_A,*h_B,*h_C;
    // vectAdd and length(vec) = 10000
    int block_len = 1024;
    int num_block = 1024*16;
    int len = block_len*num_block;
    size_t size = len*sizeof(float);
    // malloc host mem use cudaMallocHost
    cudaMallocHost((void **)&h_A, size);
    cudaMallocHost((void **)&h_B, size);
    cudaMallocHost((void **)&h_C, size);
//    h_A = (float *)malloc(size);
//    h_B = (float *)malloc(size);
//    h_C = (float *)malloc(size);

    //init vector A,B,C
    for (int i=0;i<len;i++)
    {

//        memset((h_A+i),i,sizeof(float));
//        memset((h_B+i),i,sizeof(float));
        h_A[i] = i;
        h_B[i] = 2*i;
        h_C[i] = 0;
    }
//    memset(h_C,0,size);
    jl_eval_string("println(\"julia: host A,B,C init successfully\")");
   
    // alloc device mem
    float *d_A,*d_B,*d_C;
    cudaMalloc((void **)&d_A,size);
    cudaMalloc((void **)&d_B,size);
    cudaMalloc((void **)&d_C,size);

    /*
    for (int i=0;i<len;i++)
    {
        printf("h_B[%d] is %f\n",i,h_B[i]);
    }
    */

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // memcpy ==> vec_add<<<>>> ==> memcpy #TODO stream and event async parallel maxthroughtput
    // data transfer from host to device
    cudaDeviceSynchronize();
    cudaEventRecord(start,0);
    cudaMemcpyAsync(d_A,h_A,size,cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(d_B,h_B,size,cudaMemcpyHostToDevice,0);

    dim3 dimGrid(num_block,1);
    dim3 dimBlock(block_len,1);
    vec_add<<<dimGrid,dimBlock,0,0>>>(d_A,d_B,d_C,len);
//    cudaDeviceSynchronize();

    cudaMemcpyAsync(h_C,d_C,size,cudaMemcpyDeviceToHost,0); // cudaMemcpy is a synchronize function
    cudaDeviceSynchronize();
    cudaEventRecord(stop,0);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time,start,stop);
    printf("gpu_time is %f\n",gpu_time);
    // check result 
    bool ret = true;
    for (int i=0;i<len;i++)
    {
        if (h_C[i] != 3*i)
        {   
           // printf("h_C[%d] is %f",i,h_C[i]);
            ret = false;
        }
    }
    
    ret ? printf("result is passed\n") : printf("result is failed\n");

    //Free objects
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    jl_atexit_hook(0);
    cudaDeviceReset();
    return 0;
}
