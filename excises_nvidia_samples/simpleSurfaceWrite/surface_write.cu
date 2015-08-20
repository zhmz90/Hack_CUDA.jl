#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>


__global__ void surface_write(float *g_idata, int width, int height)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    surf2Dwrite(g_idata[y*width+x], outputSurface, x*4, y, cudaBoundaryModeTrap);

}

__global__ void transform(float *g_odata, int width, int height, float theta)
{
    unsigned int gtx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int gty = blockIdx.y*blockDim.y + threadIdx.y;

    float u = x / (float)width;
    float v = y / (float)height;
    
    u -= 0.5f;
    v -= 0.5f;

    float tu = u*cosf(theta) - v*sinf(theta) + 0.5f;
    float tv = v*cosf(theta) + u*sinf(theta) + 0.5f;

    g_odata[y*width+x] = tex2D(tex,tu,tv);
}

void run_test()
{
    float *h_data,*h_data_ref,*d_data;
    
    unsigned int size = width * height * sizeof(float);
    h_data_ref = (float *)malloc(size);
    cudaMalloc((void **)&d_data,size);
    
    cudaChannelFormatDesc channelDesc = 
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    cudaArray *cuArray;
    cudaMallocArray(&cuArray,&channelDesc,width,height,cudaArraySurfaceLoadStore);

    dim3 dim_block(8,8);
    dim3 dim_grid(width/dim_block.x, height/dim_block.y);
    
    cudaMemcpy(d_data,h_data,size,cudaMemcpyHostToDevice);
    cudaBindSurfaceToArray(outputSurface,cuArray);

    surface_write<<<dim_grid,dim_block>>>(d_data,width,height);

    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = true;

    cudaBindTextureToArray(tex, cuArray, channelDesc);

    


