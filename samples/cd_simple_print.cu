#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <helper_cuda.h>
#include <helper_string.h>

__device__ int g_uids = 0;

__device__ void print_info(int depth, int thread, int uid, int parent_uid)
{
    if (threadIdx.x == 0)
    {
        if (depth == 0)
            printf("BLOCK %d launched by the host\n", uid);
        else
        {
            char buffer[32];
            
            for (int i=0; i < depth; i++)
            {
                bufferp[3*i+0] = '|';
                bufferp[3*i+1] = ' ';
                bufferp[3*i+2] = ' ';
           
            }
            buffer[3*depth] = '\0';
           // printf();
        }
    }

    __syncthreads();

}

__global__ void cdp_kernel(int max_depth, int depth, int thread, int parent_uid)
{
    __shared__ int s_uid;
    
    if (threadIdx.x == 0)
    {
        s_uid = atomicAdd(&g_uids, 1); 
    }
    __syncthreads();

    print_info(depth, thread, s_uid, parent_uid);

    if (++depth >= max_depth)
    {
        return;
    }
    cdp_kernel<<gridDim.x, blockDim.x>>>(max_depth, depth, threadIdx.x, s_uid);
}


