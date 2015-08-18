#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/mman.h> // mmap() munmap()

#include <julia.h>
#include <cuda_runtime.h>


#define MEMORY_ALIGNMENT 4096
#define ALIGN_UP(x, size) ( ((size_t)x+(size-1)&(~(size-1))) ) // ????

/*
const char *sEventSyncMethod[] =
{
    "cudaEventDefault",
    "cudaEventBlockingSync".
    "cudaEventDisableTiming",
    "NULL"
};

const char *sDeviceSyncMethod[] = 
{
    "cudaDeviceScheduleAuto",
    "cudaDeviceScheduleSpin",
    "cudaDeviceScheduleYield",
    "INVALID",
    "cudaDeviceScheduleBlockingSync",
    NULL
};
*/

// why int *factor not int?
__global__ void init_array(int *g_data, int *factor, int num_iterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i=0; i<num_iterations; i++)
    {
        g_data[idx] += *factor; // non-coalesced 
    }

}

// a[] must all be c
bool correct_data(int *a, const int n, const int c)
{   
    for (int i=0;i<n;i++)
    {
        if (a[i]!=c)
        {
            printf("correct_data is wrong\n");
            return false;
        }
    }
    return true;
}


// ??? what is the difference between mmap+cudaHostRegister
inline void alloc_host_mem(bool b_pin_generic_mem, int **pp_a, int **pp_aligned_a, int nbytes)
{
    if (b_pin_generic_mem)
    {
        *pp_a = (int *)mmap(NULL, (nbytes+MEMORY_ALIGNMENT), PROT_READ|PROT_WRITE, 
                            MAP_PRIVATE|MAP_ANON, -1, 0);

        *pp_aligned_a = (int *)ALIGN_UP(*pp_a, MEMORY_ALIGNMENT);
    
        cudaHostRegister(*pp_aligned_a, nbytes, cudaHostRegisterMapped);
    }
    else
    {
        cudaMallocHost((void **)pp_a, nbytes);
        *pp_aligned_a = *pp_a;
    }

}
inline void free_host_mem(bool b_pin_generic_mem, int **pp_a, int **pp_aligned_a, int nbytes)
{
    if (b_pin_generic_mem)
    {
        cudaHostUnregister(*pp_aligned_a);
        munmap(*pp_a, nbytes);
    }
    else
    {
        cudaFreeHost(*pp_a);
    }
}


int main(void)
{
    jl_init("/home/guo/julia/usr/lib");

    int dev_count;
    cudaGetDeviceCount(&dev_count);

    jl_function_t *func = jl_get_function(jl_base_module,"println");
    jl_value_t* num_dev = jl_box_int32(dev_count);
    jl_call1(func,num_dev);

    int cuda_device = 0;
    int nstreams = 4;
    int nreps = 10;
    int n = 16*1024*1024;
    int nbytes = n * sizeof(int);
    dim3 dim_block, dim_grid;
    float elapsed_time, time_memcpy, time_kernel;
    float scale_factor = 1.0f;

    bool b_pin_generic_mem = true;

    int device_sync_method = cudaDeviceBlockingSync;

    int niterations=5;

    cudaSetDevice(cuda_device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, cuda_device);
    printf("%s:%d.%d\n",deviceProp.name,deviceProp.major,deviceProp.minor);
    printf("deviceProp.canMapHostMemory:%d\n",(int)deviceProp.canMapHostMemory);

    scale_factor = 32.0f / 3072;

    n = (int)rint((float)n/scale_factor);

    printf("scale_factor: %1.4f\n",1.0f/scale_factor);
    printf("array_size: %d\n\n",n);

    //using CPU/GPU device sync, blocking sync to reduce CU uage
    cudaSetDeviceFlags(device_sync_method | cudaDeviceMapHost );

    // allocate host mem
    int h_c = 5;
    int *h_a = 0;
    int *h_aligned_a = 0;

    alloc_host_mem(b_pin_generic_mem,&h_a,&h_aligned_a,nbytes);

    // alloc device mem
    int *d_a = 0, *d_c = 0;
    cudaMalloc((void **)&d_a, nbytes);
    cudaMalloc((void **)&d_c, nbytes);
    cudaMemcpy(d_c, &h_c, sizeof(int),cudaMemcpyHostToDevice);

    cudaStream_t *streams = (cudaStream_t *)malloc(nstreams*sizeof(cudaStream_t));
    
    for (int i=0;i<nstreams;i++)
    {
        cudaStreamCreate(&(streams[i])); //cudaStreamCreate((streams+i));
    }

    // event handles 
    // use blocking sync
    cudaEvent_t start_event, stop_event;
    int event_flags = cudaEventBlockingSync;
    cudaEventCreateWithFlags(&start_event,event_flags);
    cudaEventCreateWithFlags(&stop_event,event_flags);

    cudaEventRecord(start_event,0);
    cudaMemcpyAsync(h_aligned_a,d_a,nbytes,cudaMemcpyDeviceToHost,streams[0]);
    cudaEventRecord(stop_event,0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&time_memcpy,start_event,stop_event);
    printf("memcpy:\t%.2f\n",time_memcpy);
    
    // time kernel
    dim_block = dim3(512,1);
    dim_grid  = dim3(n/dim_block.x,1);
    cudaEventRecord(start_event,0);
    init_array<<<dim_grid, dim_block, 0, streams[0]>>>(d_a,d_c,niterations);
    cudaEventRecord(stop_event,0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&time_kernel,start_event,stop_event);
    printf("kernel:\t\t%.2f\n",time_kernel);

    //////////////////////////////////////
    printf( "#################time non-stream exec##############\n");
    dim_block = dim3(512,1);
    dim_grid  = dim3(n/dim_block.x,1); 
    cudaEventRecord(start_event,0);

    for (int k=0; k<nreps;k++)
    {
        init_array<<<dim_grid,dim_block>>>(d_a,d_c,niterations);
        cudaMemcpy(h_aligned_a, d_a, nbytes,cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop_event,0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
    printf("non-stream:\t%.2f\n",elapsed_time/nreps);

    printf( "\n#################time with nstreams streams exec##############\n");
    dim_block = dim3(512,1);
    dim_grid  = dim3(n/dim_block.x,1);
    memset(h_aligned_a,255,nbytes);
    cudaMemset(d_a,0,nbytes);
    cudaEventRecord(start_event,0);
    
    for (int k=0;k<nreps;k++)
    {
        for (int i=0;i<nstreams;i++)
        {
            init_array<<<dim_grid,dim_block,0,streams[i]>>>(d_a+i*n/nstreams,d_c,niterations);
        }
        
        for (int i=0;i<nstreams;i++)
        {
            cudaMemcpyAsync(h_aligned_a+i*n/nstreams,d_a+i*n/nstreams,nbytes/nstreams, cudaMemcpyDeviceToHost,streams[i]);
        }
    }
    cudaEventRecord(stop_event,0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time,start_event,stop_event);
    printf("time with nstreams streams exec %f\n", elapsed_time/nreps); 

    for (int i=0;i<nstreams;i++)
    {
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    free_host_mem(b_pin_generic_mem, &h_a, &h_aligned_a, nbytes);

    cudaFree(d_a);
    cudaFree(d_c);

    cudaDeviceReset();
    jl_atexit_hook(0);

    return 0;
}
