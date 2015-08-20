#include <stdio.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define MAX(a,b) ( a>b ? a : b)

__global__ void  vector_add(float *a, float *b, float *c, int N)
{
    int gtid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gtid < N)
    {
        c[gtid] = a[gtid] + b[gtid];
    }

}

bool bPinGenericMemory = false;

#define MEMORY_ALIGNMENT 4096
#define ALIGN_UP(x,size) ( ((size_t)x + (size-1))&(~(size-1)) ) // ???

int main(int argc, char **argv)
{
    int n, nelem, deviceCount;
    int idev = 0;
    char *device = NULL;
    unsigned int flags;
    size_t bytes;
    float *a, *b, *c;
    float *a_UA, *b_UA, *c_UA;
    float *d_a, *d_b, *d_c;
    float errorNorm, refNorm, ref, diff;
    cudaDeviceProp deviceProp;

    bPinGenericMemory = true;
    
    cudaSetDevice(idev);

    cudaSetDeviceFlags(cudaDeviceMapHost);

    nelem = 1048576;
    bytes = nelem*sizeof(float);
    
    if (bPinGenericMemory)
    {
        a_UA = (float *)malloc(bytes + MEMORY_ALIGNMENT);
        b_UA = (float *)malloc(bytes + MEMORY_ALIGNMENT);
        c_UA = (float *)malloc(bytes + MEMORY_ALIGNMENT);
  
        a = (float *) ALIGN_UP(a_UA, MEMORY_ALIGNMENT);
        b = (float *) ALIGN_UP(b_UA, MEMORY_ALIGNMENT);
        c = (float *) ALIGN_UP(c_UA, MEMORY_ALIGNMENT);

        cudaHostRegister(a, bytes, CU_MEMHOSTALLOC_DEVICEMAP);
        cudaHostRegister(b, bytes, CU_MEMHOSTALLOC_DEVICEMAP);
        cudaHostRegister(c, bytes, CU_MEMHOSTALLOC_DEVICEMAP);
    }
    else
    {
        flags = cudaHostAllocMapped;
        cudaHostAlloc((void **)&a, bytes, flags);
        cudaHostAlloc((void **)&b, bytes, flags);
        cudaHostAlloc((void **)&c, bytes, flags);
    }

    for (n=0;n<nelem;n++)
    {
        a[n] = rand() / (float)RAND_MAX;
        b[n] = rand() / (float)RAND_MAX;
    }

    cudaHostGetDevicePointer((void **)&d_a, (void *)a, 0);
    cudaHostGetDevicePointer((void **)&d_b, (void *)b, 0);
    cudaHostGetDevicePointer((void **)&d_c, (void *)c, 0);

    dim3 dim_block(256);
    dim3 dim_grid((unsigned int)ceil(nelem/(float)dim_block.x));
    vector_add<<<dim_block,dim_grid>>>(d_a,d_b,d_c,nelem);
    cudaDeviceSynchronize();
    
    if (bPinGenericMemory)
    {
        cudaHostUnregister(a);
        cudaHostUnregister(b);
        cudaHostUnregister(c);
        free(a_UA);
        free(b_UA);
        free(c_UA);
    }
    else
    {
        cudaFreeHost(a);
        cudaFreeHost(b);
        cudaFreeHost(c);
  
    }

    cudaDeviceReset();
    return 0;
}











