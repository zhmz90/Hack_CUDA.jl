#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <julia.h>
#include <cuda_runtime.h>

__global__ void add_constant(int *d_a, const int b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_a[idx] += b;
}

int main(int argc, char *argv[])
{
    
    printf("%s Starting...\n\n",argv[0]);


    printf("omp_get_num_procs() is %d\n",omp_get_num_procs());
    int num_gpu = 2;
    unsigned int n = num_gpu * 8192;
    unsigned int nbytes = n * sizeof(int);
    int *a = 0;
    int b = 3;
    a = (int *)malloc(nbytes);
    
    for (unsigned int i=0;i<n;i++)
    {
        a[i] = i;
    }

    omp_set_num_threads(10);
    #pragma omp parallel
    {
        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();
    
        int gpu_id = -1;
        cudaSetDevice(cpu_thread_id % 2);
        cudaGetDevice(&gpu_id);
        printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id,num_cpu_threads,gpu_id);
        
        int *d_a = 0;
        int *sub_a = a + cpu_thread_id * n/num_cpu_threads;
        
        unsigned int nbytes_per_kernel = nbytes / num_cpu_threads;
        
        dim3 dim_block(128);
        dim3 dim_grid(n/dim_block.x);
    
        cudaMalloc((void **)&d_a,nbytes_per_kernel);
        cudaMemcpy(d_a,sub_a,nbytes_per_kernel,cudaMemcpyHostToDevice);
        add_constant<<<dim_grid,dim_block>>>(d_a,b);
        cudaMemcpy(sub_a,d_a,nbytes_per_kernel,cudaMemcpyDeviceToHost);
        cudaFree(d_a);
    }

    if (cudaSuccess != cudaGetLastError())
        printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    if (a)
        free(a);

    cudaDeviceReset();
    return 0;
}

