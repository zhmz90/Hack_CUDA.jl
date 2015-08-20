#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>

#define MAX_EPSILON_ERROR 5e-3f

const char *imageFilename = "lena_bw.pgm";
const char *refFilename = "ref_rotated.pgm";
const char *sampleName = "simpleTexture";
const float angle = 0.5f;

texture<float, 2, cudaReadModeElementType> tex;

__global__ void transform_kernel(float *outputData, int width, int height, float theta)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float u = (float)x - (float)width/2;
    float v = (float)y - (float)height/2;
    float tu = u*cosf(theta) - v*sinf(theta);
    float tv = v*cosf(theta) + v*sinf(theta);

    tu /= (float)width;
    tv /= (float)height;

    outputData[y*width+x] = tex2D(tex, tu+0.5f, tv+0.5f);
}

void run_test(int argc, char **argv);

int main(int argc, char **argv)
{
    run_test(argc, argv);

    cudaDeviceReset();
    return 0;
}

void run_test(int argc, char **argv)
{
    int devID = 1;
    
    float *h_data = NULL;
    unsigned int width, height;
    const char *image_path = 
        "/home/guo/haplox/GPU/NVIDIA_CUDA-7.0_Samples/0_Simple/simpleTexture/data/lena_bw.pgm"

    sdkLoadPGM(image_path, &h_data, &width, &height);

    unsigned int size = width * height * sizeof(float);

    printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

    float *h_data_ref = (float *)malloc(size);
    const char *ref_path = 
         "/home/guo/haplox/GPU/NVIDIA_CUDA-7.0_Samples/0_Simple/simpleTexture/data/ref_rotated.pgm"
    sdkLoadPGM(&ref_path, &h_data_ref,&width, &height);

    float *d_data = NULL;
    cudaMalloc((void **)&d_data, size);

    cudaChannelFormatDesc channelDesc = 
        cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
    cudaArray *cuArray;
    cudaMallocArray(&cuArray,&channelDesc,width,height);
    cudaMemcpyToArray(cuArray,0,0,h_data,size,cudaMemcpyHostToDevice);
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = true;
    cudaBindTextureToArray(tex,cuArray,channelDesc);

    dim3 dim_block(8,8);
    dim3 dim_grid(width/dim_block.x, height/dim_block.y);

    transform_kernel<<<dim_grid, dim_block>>>(d_data, width, height, angle);
    cudaDeviceSynchronize();
    
    float *h_output_data = (float *) malloc(size);
    cudaMemcpy(h_output_data, d_data, size, cudaMemcpyDeviceToHost);
    printf("%d\t%d\t%d\n", h_output_data[0],h_output_data[1],h_output_data[2]);

    cudaFree(d_data);
    cudaFreeArray(cuArray);
    free(image_path);
    free(ref_path);
}
