// ??? C/C++ headers can be loaded in the same file
#include <stdio.h>
#include <iostream> 

#include <vector>
#include <cuda_runtime.h>

#include "device_library.cuh"

using std::cout;
using std::endl;
using std::vector;


#define EPS 1e-5

typedef unsigned int uint;
typedef float(*deviceFunc)(float); //######### define a functional pointer

__device__ deviceFunc d_multi2 = multiplyByTwo;
__device__ deviceFunc d_div2   = divideByTwo;


__global__ void transform(float *vec, deviceFunc f, uint size)
{
    uint gt = blockDim.x*blockIdx.x + threadIdx.x;
    if (gt < size)
    {
        vec[gt] = (*f)(vec[gt]);
    }
}

void run_test(int argc, const char **argv);

int main(int argc, char **argv)
{
    run_test(argc, (const char **)argv);

    cout<<"seems ok"<<endl;
    return 0;
}

void run_test(int argc, const char **argv)
{
    try
    {
        const uint vec_len = 1000;
        
        vector<float> vec(vec_len);

        for (uint i=0; i < vec_len; i++)
        {
            vec[i] = rand() / static_cast<float>(RAND_MAX);
        }
    
        float *d_vec;
        cudaMalloc(&d_vec,vec_len*sizeof(float));
        cudaMemcpy(d_vec,&vec[0],vec_len*sizeof(float),
                cudaMemcpyHostToDevice);

        dim3 dim_block(1024);
        dim3 dim_grid(1);
    
        deviceFunc h_funcptr;
        
        cudaMemcpyFromSymbol(&h_funcptr, d_multi2, sizeof(deviceFunc));
        transform<<<dim_grid,dim_block>>>(d_vec, h_funcptr, vec_len);

        vector<float> ret_vec(vec_len);
        cudaMemcpy(&ret_vec[0],d_vec,vec_len*sizeof(float),cudaMemcpyDeviceToHost);    
    
        cudaFree(d_vec);


    }
    catch (...)
    {
        cout<<"err"<<endl;
    
    }

}

