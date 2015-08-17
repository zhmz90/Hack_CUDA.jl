#include <stdio.h>
#include <stdlib.h>

#include <julia.h>

#include <cuda_runtime.h>


typedef struct 
{
    // host-side
    int dataN;
    float *h_Data;

    //Partial sum for this GPU
    float *h_Sum;

    //Device buffers
    float *d_Data,*d_Sum;

    //Reduction 
    float *h_Sum_from_device;

    //Stream for async
    cudaStream_t  stream;

} TGPUplan;



__global__ muti_GPU()
{


}



int main(void)
{
    


    return 0;
}
