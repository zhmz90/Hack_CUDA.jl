#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#include "helper_h.h"

__global__ void increment_kernel(int *g_data, int inc_value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + inc_value; // may overflow

}

bool correct_output(int *data, const int n, const int x)
{
    for (int i=0;i<n;i++)
    {
        if (data[i] != x)
        {
            printf("error in correct_output\n");
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[])
{
    int devID;
    cudaDeviceProp deviceProps;
//    devID = findCudaDevice(argc, (char **)argv);
    devID = 0;
    check_error(cudaGetDeviceProperties(&deviceProps, devID));

    printf("CUDA device [%s] \n", deviceProps.name);

    int n = 16*1024*1024;
    int nbytes = n*sizeof(int);

    int value = 26;

    // alloc host mem
    int *h_a = 0;
    check_error(cudaMallocHost((void **)&h_a,nbytes));
    memset(h_a, 0, nbytes);

    // alloc devie
    int *d_a = 0;
    check_error(cudaMalloc((void **)&d_a, nbytes));
    check_error(cudaMemset(d_a,0,nbytes));

    dim3 dim_block(512,1);
    dim3 dim_grid(n/dim_block.x,1);

    cudaEvent_t start, stop;
    check_error(cudaEventCreate(&start));
    check_error(cudaEventCreate(&stop));
/*
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
*/
    check_error(cudaDeviceSynchronize());

    float gpu_time = 0.0f;
//    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);
    cudaMemcpyAsync(d_a,h_a,nbytes,cudaMemcpyHostToDevice, 0);
    increment_kernel<<<dim_grid,dim_block>>>(d_a, value);
    cudaMemcpyAsync(h_a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(stop, 0);
//    sdkStopTimer(&timer);

    // cudaEventQuery 
    float counter=0;
    while (cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter++;
    }
    check_error(cudaEventElapsedTime(&gpu_time, start, stop));

    printf("gpu_time is %f\n",gpu_time);
//   printf("sdkGetTimerValue(&timer is %f\n)",sdkGetTimerValue(&timer))
    printf("counter is %f\n",counter);

    bool cpu_gpu = correct_output(h_a, n ,value);

    check_error(cudaEventDestroy(start));
    check_error(cudaEventDestroy(stop));
    check_error(cudaFreeHost(h_a));
    check_error(cudaFree(d_a));

    cudaDeviceReset();
    exit(cpu_gpu ? EXIT_SUCCESS : EXIT_FAILURE);
}
