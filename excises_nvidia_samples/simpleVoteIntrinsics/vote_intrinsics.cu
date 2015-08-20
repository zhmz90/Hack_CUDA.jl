#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>
#include "vote_kernel.cuh"

#define MAX(a,b) (a>b ? a : b)
#define VOTE_DATA_GROUP 4

__global__ void vote_any_kernel1(unsigned int *input, unsigned int *result, int size)
{
    int tx = threadIdx.x;
    result[tx] = any(input[tx]);
}
__global__ void vote_any_kernel2(unsigned int *input, unsigned int *result, int size)
{
    int tx = threadIdx.x;
    result[tx] = all(input[tx]);
}
__global__ void vote_any_kernel3(bool *info, int warp_size)
{
    int tx = threadIdx.x;
    bool *offs = info + (tx * 3);

    *offs = any((tx>=(warp_size*3)/2));
    *(offs+1) = (tx >= (warp_size*3)/2 ? true : false);
    
    if (all((tx >= (warp_size*3)/2 ? true : false)))
    {
        *(offs + 2) = true;
    }

}

__global__ void test(void)
{
    printf("warp_size: %d\n",warpSize);
}

int main()
{
    
    test<<<1,1>>>();
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}
