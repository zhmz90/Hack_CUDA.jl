#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <julia.h>
#include <cuda_runtime.h>


#define NUM_BLOCKS 512 //64
#define NUM_THREADS 256


// find min(input[])
// static can be replaced by inline
__global__ static void reduction(const float *input, float *output, clock_t *timer)
{
    extern __shared__ float shared[];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (tid == 0) timer[bid] = clock();

    shared[tid] = input[tid];
    shared[tid + blockDim.x] = input[tid+blockDim.x];
    
    for (int d = blockDim.x; d>0;d/=2)
    {
        __syncthreads();

        if (tid < d)
        {
            float f0 = shared[tid];
            float f1 = shared[tid + d];
        
            if (f1 < f0)
            {
                shared[tid] = f1;
            }
        }
    }

    if (tid == 0) output[bid] = shared[0];

    __syncthreads();

    if (tid == 0) timer[bid+gridDim.x] = clock();

}



int main(void)
{
    float *d_input = NULL;
    float *d_output = NULL;
    clock_t *d_timer = NULL;

    clock_t h_timer[NUM_BLOCKS * 2];
    float h_input[NUM_THREADS * 2];

    for (int i=0; i < NUM_THREADS*2; i++)
    {
        h_input[i] = (float)i;
    }
    
    size_t size = sizeof(float)*NUM_THREADS;
    cudaMalloc((void **)&d_input, size*2);
    cudaMalloc((void **)&d_output, size);
    cudaMalloc((void **)&d_timer, sizeof(clock_t)*NUM_THREADS*2);
    cudaMemcpy(d_input,h_input,size*2,cudaMemcpyHostToDevice);
    reduction<<<NUM_BLOCKS,NUM_THREADS,sizeof(float)*2*NUM_THREADS>>>(d_input,d_output,d_timer);
    cudaMemcpy(h_timer,d_timer,sizeof(clock_t)*NUM_BLOCKS*2,cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_timer);

    clock_t min_start = h_timer[0];
    clock_t max_end   = h_timer[NUM_BLOCKS];
    for (int i=1;i<NUM_BLOCKS;i++)
    {
        min_start = h_timer[i] < min_start ? h_timer[i]:min_start;
        max_end = h_timer[NUM_BLOCKS+i] > max_end ? h_timer[NUM_BLOCKS+i]:max_end;
    }

    printf("Total clock time exec by kernel:%d\n", (int)(max_end-min_start));


    cudaDeviceReset();
    return 0;
}
