#include <stdio.h>
#include <stdlib.h>

#include <julia.h>

#include <cuda_runtime.h>

#define MAX(a,b) (a > b ? a : b)


__global__ void print_info(int val)
{
    printf("[%d,%d]:\t\tValue is:%d\n",\
            blockIdx.y*gridDim.x+blockIdx.x,\ //the Grid is two dimensional
            threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\ // block is three dim
            val);
}

int main(void)
{
    jl_init("/home/guo/julia/usr/lib");
        
    int devID = 1;
    cudaDeviceProp props;

    cudaGetDevice(&devID);
    cudaGetDeviceProperties(&props,devID);
    
    printf("Device %d: %s, %d.%d\n",devID,props.name,props.major,props.minor);

    dim3 dimGrid(2,2);
    dim3 dimBlock(1,1,1);
    
    print_info<<<dimGrid,dimBlock>>>(12);
    cudaDeviceSynchronize();
    
    cudaDeviceReset();
    jl_atexit_hook(0);

    return 0;
}

