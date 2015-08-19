#include <stdio.h>
#include <stdlib.h>

#include <julia.h>
#include <cuda_runtime.h>

#include "multithreading.h"

const int N_workloads = 8;
const int N_elements_per_workload = 100000;

CUTBarrier thread_barrier;

void CUDART_CB myStreamCallback(cudaStream_t event, cudaError_t status, void *data);

struct heter_workload
{
    int id;
    int cudaDeviceID;
    
    int *h_data;
    int *d_data;
    cudaStream_t stream;

    bool success;
};

__global__ void add_one(int *data, int N)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%d\n",gtid);

    if (gtid < N)
        data[gtid]++;

}

CUT_THREADPROC launch(void *void_arg)
{
    heter_workload *workload = (heter_workload *) void_arg;
    cudaSetDevice(workload->cudaDeviceID);

    cudaStreamCreate(&workload->stream);
    cudaMalloc((void **)&workload->d_data, N_elements_per_workload*sizeof(int));
    cudaHostAlloc((void **)&workload->h_data, N_elements_per_workload*sizeof(int), cudaHostAllocPortable);
   
    for (int i=0; i< N_elements_per_workload; i++)
    {
        workload->h_data[i] = workload->id + i;
    }

    dim3 dim_block(512);
    dim3 dim_grid((N_elements_per_workload + dim_block.x -1) / dim_block.x);
    
    cudaMemcpyAsync(workload->d_data, workload->h_data, N_elements_per_workload*sizeof(int), 
            cudaMemcpyHostToDevice, workload->stream);
    add_one<<<dim_grid,dim_block>>>(workload->d_data, N_elements_per_workload);
    cudaDevieSynchronize();
    cudaMemcpyAsync(workload->h_data, workload->d_data, N_elements_per_workload*sizeof(int),
            cudaMemcpyDeviceToHost, workload->stream);

    // add a cpu callback which is called once all currently pending operation in the cuda stream have finished
    cudaStreamAddCallback(workload->stream, myStreamCallback,workload,0);

    CUT_THREADEND;
    // cpu thread end of life, GPU continues.... ???
}

CUT_THREADPROC postprocess(void *void_arg)
{
    heter_workload *workload = (heter_workload *) void_arg;
    cudaSetDevice(workload->cudaDeviceID);
    workload->success = true;
    
    for (int i=0;i<N_workloads;i++)
    {
        workload->success &= workload->h_data[i] == i + workload->id + 1;
    }

    cudaFree(workload->d_data);
    cudaFreeHost(workload->h_data);
    cudaStreamDestroy(workload->stream);

    // signal the end of heter to main thread
    cutIncrementBarrier(&thread_barrier);

    CUT_THREADEND;
}

// the para ### ??? 
void CUDART_CB myStreamCallback(cudaStream_t stream, cudaError_t status, void *data)
{
    cutStartThread(postprocess, data); // ???
}


int main(void)
{
    int N_gpus;
    
    cudaGetDeviceCount(&N_gpus);

    heter_workload *workloads;
    workloads = (heter_workload *) malloc(N_workloads * sizeof(heter_workload));
    thread_barrier = cutCreateBarrier(N_workloads);

    for (int i=0; i<N_workloads; i++)
    {
        workloads[i].id = i;
        workloads[i].cudaDeviceID = i % 2;

        cutStartThread(launch, &workloads[i]);
    }

    cutWaitForBarrier(&thread_barrier);
    printf("%d workloads finished\n", N_workloads);

    bool success = true;
    for (int i=0; i<N_workloads;i++)
    {
        success &= workloads[i].success;
    }
    printf("%s\n",success ? "success":"Failure");


    cudaDeviceReset();
    
    free(workloads);

    return 0;
}


