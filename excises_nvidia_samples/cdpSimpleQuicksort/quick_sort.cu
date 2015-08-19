#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <julia.h>
#include <cuda_runtime.h>

#define MAX_DEPTH 16
#define INSERTION_SORT 32


// each time select a smallist val insert the first
__device__ void selection_sort(unsigned int *data, int left, int right)
{
    for (int i=left; i<= right; i++)
    {
        unsigned min_val = data[i];
        int min_idx = i;

        // find the smallist val in left and right
        for (int j=i+1;j<=right;j++)
        {
            unsigned val_j = data[j];
        
            if (val_j < min_val)
            {
                min_idx = j;
                min_val = val_j;
            }
        }
        if (i != min_idx)
        {
            data[min_idx] = data[i];
            data[i] = min_val;
        }

    }

}

__global__ __device__ void quick_sort(unsigned int *data, int left, int right, int depth)
{
    if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT)
    {
        selection_sort(data, left, right);
        return;
    }

    unsigned int *lptr = data + left;
    unsigned int *rptr = data + right;
    unsigned int pivot = data[(left+right)/2];
    
    // Partition the vector
    while (lptr <= rptr)
    {
        unsigned int lval = *lptr;
        unsigned int rval = *rptr;

        while (lval < pivot)
        {
            lptr++;
            lval = *lptr;
        }
        while (rval > pivot)
        {
            rptr--;
            rval = *rptr;
        }
        if (lptr <= rptr)
        {
            *lptr++ = rval;
            *rptr-- = lval;
        }
    }
    
    // Now the recursive part
    int nright = rptr - data;
    int nleft  = lptr - data;
    
    // Lauch a new block to sort the left part.
    if (left < (rptr - data))
    {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        quick_sort<<<1,1,0,s>>>(data,left,nright,depth+1);
        cudaStreamDestroy(s);
    }

     // Lauch a new block to sort the right part.
    if (right > (lptr-data))
    {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        quick_sort<<<1,1,0,s>>>(data,nleft,right,depth+1);
        cudaStreamDestroy(s);
    }
}

void run_qsort(unsigned int *data, unsigned int nitems)
{
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, (size_t)MAX_DEPTH);

    int left = 0;
    int right = nitems - 1;
    std::cout << "Launching kernel on the GPU" << std::endl;
    quick_sort<<<1,1>>>(data,left,right,0);
    cudaDeviceSynchronize();
}

void initialize_data(unsigned int *dst, unsigned int nitems)
{
    srand(1);

    for (unsigned i = 0; i < nitems;i++)
        dst[i] = rand() % nitems;

}


int main(void)
{
    // init julia
    jl_init("/home/guo/julia/usr/lib");
    jl_eval_string("println(\"julia: julia init successfully\")");


    cudaSetDevice(1);
    
    unsigned int *h_data = 0;
    unsigned int *d_data = 0;

    std::cout << "Initializing data:" << std::endl;

    int num_items = 128;
    h_data = (unsigned int *)malloc(num_items*sizeof(unsigned  int));
    initialize_data(h_data, num_items);

    cudaMalloc((void **)&d_data, num_items*sizeof(unsigned  int));
    cudaMemcpy(d_data, h_data, num_items*sizeof(unsigned  int), cudaMemcpyHostToDevice);

    run_qsort(d_data, num_items);

    free(h_data);
    cudaFree(d_data);

    jl_atexit_hook(0);
    cudaDeviceReset();
    return 0;
}
