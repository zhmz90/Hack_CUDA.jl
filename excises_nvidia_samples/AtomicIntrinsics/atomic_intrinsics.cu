#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <julia.h>
#include <cuda_runtime.h>

__global__ void test_kernel(int *g_odata)
{
    const unsigned int gt = blockDim.x*blockIdx.x+ threadIdx.x;

    atomicAdd(&g_odata[0],10);
    atomicSub(&g_odata[1],10);
    atomicExch(&g_odata[2],gt);
    atomicMax(&g_odata[3],gt);
    atomicMin(&g_odata[4],gt);
    atomicInc((unsigned int *)&g_odata[5],17);
    atomicDec((unsigned int *)&g_odata[6],137);
    atomicCAS(&g_odata[7],gt-1, gt);
    atomicAdd(&g_odata[8],2*gt+7);
    atomicOr(&g_odata[9],1<<gt);
    atomicXor(&g_odata[10],gt);

}


void run_test(int argc, char **argv)
{
    unsigned int dim_block(256);
    unsigned int dim_grid(64);
    unsigned int numData = 11;
    unsigned int memSize = sizeof(int)*numData;

    int *h_odata = (int *)malloc(memSize);

    for (unsigned int i=0;i<numData;i++)
    {
        h_odata[i]=0;
    }
    
    h_odata[8] = h_odata[10] = 0xff;

    int *d_odata;
    cudaMalloc((void **)&d_odata,memSize);
    cudaMemcpy(d_odata,h_odata,memSize,cudaMemcpyHostToDevice);
    test_kernel<<<dim_grid,dim_block>>>(d_odata);
    cudaMemcpy(h_odata,d_odata,memSize,cudaMemcpyDeviceToHost);
 
    for (int i=0;i<numData;i++)
    {
        printf("%d\t",h_odata[i]);
    }
    printf("\n");
    /*
    jl_value_t* array_type = jl_apply_array_type(jl_float64_type,1);    
    jl_array_t * h_data = jl_alloc_array_1d(array_type,10);

    JL_GC_PUSH1(&h_data);
    
    double *data = (double *)jl_array_data(h_data);
    for (int i=0;i<10;i++)
    {
        data[i] = i;
    }
//jl_eval_string("@show data");

    JL_GC_POP();
*/
    free(h_odata);
    cudaFree(d_odata);
}


int main(int argc, char **argv)
{
    //jl_init("/home/guo/julia/usr/lib");

    run_test(argc, argv);

    cudaDeviceReset();
  //  jl_atexit_hook(0);
    return 0;
}
