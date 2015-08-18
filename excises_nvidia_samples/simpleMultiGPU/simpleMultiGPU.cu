#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <julia.h>
#include <cuda_runtime.h>


#define MAX(a,b) (a>b ? a:b)

const int MAX_GPU_COUNT = 2;
const int DATA_N = 1048576*32;

typedef struct 
{
    // host-side
    int num_data;
    float *h_data;

    //Partial sum for this GPU
    float *h_sum;

    //Device buffers
    float *d_data,*d_sum;

    //Reduction 
    float *h_sum_from_device;

    //Stream for async
    cudaStream_t  stream;

} TGPUplan;



__global__ muti_GPU()
{

}

__global__ static void reduce_kernel(float *d_result, float *d_input, int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x; // threadIdx global
    const int threadN = gridDim.x * blockDim.x; // num of threads under this Grid
    float sum = 0;
    
    for (int pos = tid; pos<N; pos += threadN)
    {
        sum += d_input[pos];
    }

    d_result[tid] = sum;
}

int main(void)
{
    TGPUplan plan[MAX_GPU_COUNT];
    float h_sumGPU[MAX_GPU_COUNT];
    float sum_GPU;
    double sum_CPU,diff;

    int i,j,gpu_base,GPU_N;

    const int BLOCK_N = 32;
    const int THREAD_N = 256;
    const int ACCUM_N = BLOCK_N * THREAD_N;

    cudaGetDeviceCount(&GPU_N);
    printf("num of GPU:%d\n", GPU_N);
    
    for (i=0;i<GPU_N;i++)
    {
        plan[i].num_data = DATA_N / GPU_N; // how many data in each host GPU
    }
    for (i=0;i < DATA_N % GPU_N;i++)
    {
        plan[i].num_data++;
    }
    gpuBase = 0;
    for (i=0;i<GPU_N;i++)
    {
        plan[i].h_sum = h_sumGPU + i;
        gpuBase += plan[i].num_data;
    }

    for (i=0;i<GPU_N;i++)
    {
        cudaSetDevice(i); // set GPU device
        cudaStreamCreate(&plan[i].stream);
        
        cudaMalloc((void **)&plan[i].d_data, plan[i].num_data*sizeof(float));
        cudaMalloc((void **)&plan[i].d_sum, ACCUM_N*sizeof(float));
        cudaMallocHost((void **)&plan[i].h_data, plan[i].num_data*sizeof(float));
        cudaMallocHost((void **)&plan[i].h_sum_from_device, ACCUM_N*sizeof(float));

        for (j=0;j<plan[i].num_data;j++)
        {
            plan[i].h_data[j] = (float)rand()/(float)RAND_MAX;
        }

    }

    for (i=0;i<GPU_N;i++)
    {
        cudaSetDevice(i); // Set GPU device
        cudaMemcpyAsync(plan[i].d_data,plan[i].h_data,plan[i].num_data*sizeof(float),cudaMemcpyHostToDevice,plan[i].stream);
        reduce_kernel<<<BLOCK_N,THREAD_N,0,plan[i].stream>>>(plan[i].d_sum,plan[i].d_data,plan[i].num_data);
        //getLastCudaError("kernel failed");

        cudaMemcpyAsync(plan[i].h_sum_from_device,plan[i].d_sum,ACCUM_N,cudaMemcpyDeviceToHost,plan[i].stream);
    }

    for (i=0;i<GPU_N;i++)
    {
        float sum;
        cudaSetDevice(i);
        cudaStreamSynchronize(plan[i].stream);
        
        sum = 0;
        
        for (j=0;j<ACCUM_N;j++)
        {
            sum += plan[i].h_sum_from_device[j];
        }
        *(plan[i].h_sum) = (float)sum;
        
        cudaFreeHost(plan[i].h_sum_from_device);
        cudaFree(plan[i].h_sum);
        cudaFree(plan[i].h_data);
        cudaStreamDestroy(plan[i].stream);

    }

    for (i=0;i<GPU_N;i++)
    {
        cudaSetDevice(i);
        cudaFreeHost(plan[i].h_data);

        cudaDeviceReset();
    }


    return 0;
}
