#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// global variables
__device__ int i=0;
__device__ const int max_depth = 2;

// printinfo of a thread: block, parent_block, thread, parent_thread
__device__ void print_info2(int thread_id, int block_id)
{
    printf("thread %d from block %d \n", thread_id, block_id);
}
__device__ void print_info3(int thread_id, int block_id, int depth)
{
    printf("thread %d from block %d and depth: %d\n", thread_id, block_id, depth);
}
__device__ void print_info4(int thread_idx, int thread_idy, int block_idx, int block_idy)
{
    printf("threadIdx.x: %d thredIdx.y: %d from blockIdx.x: %d blockIdx.y: %d\n", thread_idx, \
            thread_idy, block_idx, block_idy);
}
__device__ void print_info5(int thread_idx, int thread_idy, int block_idx, int block_idy, int block_id)
{
    printf("threadIdx.x: %d thredIdx.y: %d from blockIdx.x: %d blockIdx.y: %d, block_id: %d\n", thread_idx, \
            thread_idy, block_idx, block_idy, block_id);
}
// dynamic lauch a kernel if current depth is less than max_depth
__global__ void dynamic_parallel(int depth)
{
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    
    int block_id = block_y*blockDim.x + block_x;
    int thread_id = block_id*(blockDim.x*blockDim.y) + threadIdx.x*threadIdx.y + threadIdx.x;
//    print_info4(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y);
//    print_info5(threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,block_id);
//    print_info3(thread_id, block_id, depth);
    if (depth < max_depth && threadIdx.x == 0)
    {   
        printf("lauch a new kernel by threadIdx.x 0\n");
        printf("depth is %d\n",depth);
        dynamic_parallel<<<2,2>>>(++depth);
    }
}


// main: there is no need to alloc mem
int main(void)
{
   // int max_depth = 3;
    dim3 dimBlock(1,1);
    dim3 dimGrid(1,1);
    dynamic_parallel<<<dimGrid, dimBlock>>>(0);
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}

