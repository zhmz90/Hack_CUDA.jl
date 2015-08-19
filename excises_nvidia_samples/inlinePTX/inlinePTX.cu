#include <stdio.h>
#include <assert.h>

#include <julia.h>
#include <cuda_runtime.h>


__global__ void sequence_gpu(int *d_ptr, int length)
{
    int elemID = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (elemID < length)
    {
        unsigned int laneid;
        asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
        d_ptr[elemID] = laneid;
    }

}


int main(void)
{
    jl_init("/home/guo/julia/usr/lib");
    jl_eval_string("println(\"julia init successfully\n\")");

    const int N = 1000;
    size_t size = N*sizeof(int);
    int *d_ptr,*h_ptr;

    cudaMalloc((void **)&d_ptr, size);
    cudaMallocHost((void **)&h_ptr,size);

    dim3 dim_block(64,1);
    dim3 dim_grid(N/dim_block.x+1,1);
    
    sequence_gpu<<<dim_grid, dim_block>>>(d_ptr,N);
    
    cudaMemcpy(h_ptr,d_ptr,size,cudaMemcpyDeviceToHost);

    for (int i=0;i<N;i++)
    {
        printf("%d\t",h_ptr[i]);
        if (0 == i % 10)
        {
            printf("\n");
        }
    }
    printf("\n");

    cudaFree(d_ptr);
    cudaFreeHost(h_ptr);

    jl_atexit_hook(0);
    cudaDeviceReset();
    return 0;
}
