#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/utsname.h>

#include <julia.h>
#include <cuda_runtime.h>

__global__ void test_kernel(int N)
{
    int gtid = blockIdx.x*blockDim.x+ threadIdx.x;
    assert(gtid < N);
}

void run_tests(int argc, char **argv);

int main(int argc, char **argv)
{

    run_tests(argc, argv);

    cudaDeviceReset();

    return 0;
}

void run_tests(int argc, char **argv)
{
    int Nblocks = 2;
    int Nthreads = 32;
    cudaError_t error;

    utsname os_system_type;
    uname(&os_system_type);
    printf("os_system_type.release is %s\n", os_system_type.release);
    printf("os_system_type.name is %s\n", os_system_type.sysname);
    printf("os_system_type.version is %s\n", os_system_type.version);

    dim3 dim_block(Nthreads);
    dim3 dim_grid(Nblocks);

    test_kernel<<<dim_grid,dim_block>>>(60);
    error = cudaDeviceSynchronize();

    if (error == cudaErrorAssert)
    {
        printf("%s\n",cudaGetErrorString(error));
    }
    
}

