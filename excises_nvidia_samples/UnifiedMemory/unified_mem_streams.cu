#include <cstdio>
#include <ctime>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <stdlib.h>

#include <cublas_v2.h>

template <typename T>
struct Task
{
    unsigned int size, id;
    T *data;
    T *result;
    T *vector;
    
    Task() : size(0), id(0), data(NULL), result(NULL), vector(NULL) {};
    Task(unsigned int s): size(s), id(0), data(NULL), result(NULL), vector(NULL)
    {
        cudaMallocManaged(&data, sizeof(T)*size*size);
        cudaMallocManaged(&result, sizeof(T)*size);
        cudaMallocManaged(&vector, sizeof(T)*size);
        cudaDeviceSynchronize();
    }
    
    ~Task()
    {
        cudaDeviceSynchronize();
        cudaFree(data);
        cudaFree(result);
        cudaFree(vector);
    }

    void allocate(const unsigned int s, const unsigned int unique_id)
    {
        id = unique_id;
        size = s;
        cudaMallocManged(&data, sizeof(T)*size*size);
        cudaMallocManged(&data, sizeof(T)*size);
        cudaMallocManged(&data, sizeof(T)*size);
  
        for (int i=0;i<size*size;i++)
        {
            data[i] = drand48();
        }
        for (int i=0;i<size;i++)
        {
            result[i] = 0.;
            vector[i] = drand48();
        }

    }

};

// data is row-major and square
// matrix by vector
template <typename T>
void gemv(int m, int n, T alpha, T *A, T *x, T beta, T *result)
{
    for (int i=0; i<n; i++)
    {
        result[i] *= beta;
        for (int j=0; j<n;j++)
        {
            result[i] += A[i*n+j] * x[j];
        }
    }
}

template <typename T>
void execute(Task<T> &t, cublasHandle_t *handle, cudaStream_t *stream, int tid)
{
    if (t.size < 100)
    {
        printf("t.id%d,tid.t%d,t.size%d\n", (int)t.id,tid.t,t.size);
        
        cudaStreamAttachMemAsync(stream[0], t.data, 0, cudaMemAttachHost);
        cudaStreamAttachMemAsync(stream[0], t.result, 0, cudaMemAttachHost);
        cudaStreamAttachMemAsync(stream[0], t.vector, 0, cudaMemAttachHost);
        
        cudaStreamSynchronize(stream[0]);
        gemv(t.size,t.size, 1.0, t.data, t.vector, 0.0, t.result);
    }
    else
    {
        printf("t.id%d,tid.t%d,t.size%d\n",t.id,tid.t,t.size);
        double one = 1.0;
        double zero = 0.0;

        cublasSetStream(handle[tid+1], stream[tid+1]); // ???
        cudaStreamAttachMemAsync(stream[tid+1],t.data,0,cudaMemAttachSingle);
        cudaStreamAttachMemAsync(stream[tid+1],t.vector,0,cudaMemAttachSingle);
        cudaStreamAttachMemAsync(stream[tid+1],t.result,0,cudaMemAttachSingle);
        
        cublasDgemv(handle[tid+1], CUBLAS_OP_N, t.size, t.size, &one, t.data, 
                t.size, t.vector, 1, &zero, t.result, 1);

    }
}

template <typename T>
void initialise_tasks(std::vector< Task<T> > &TaskList)
{
    for (unsigned int i=0; i< TaskList.size(); i++)
    {
        int size;
        size = std::max((int)(drand48()*1000.0), 64);
        TaskList[i].allocate(size,i);
    }

}

int main(void)
{
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 1);
    
    if (device_prop.managedMemory)
    {
        printf("mangedMemory is true\n");
    }
    else
    {
        printf("mangedMemory is false\n");
    }

    const int nthreads = 4;
    omp_set_num_threads(nthreads);
    cudaStream_t *streams = new cudaStream_t[nthreads+1];
    cublasHandle_t *handles = new cublashandle_t[nthreads+1];

    for (int i=0; i<nthreads+1;i++)
    {
        cudaStreamCreate(&streams[i]);
        cublasCreate(&handles[i]);
    }

    unsigned int N = 40;
    std::vector< Task<double> > TaskList(N);
    initialise_tasks(TaskList);

    unsigned int i;
    #pragma omp parallel for schedule(dynamic)

    for (i=0; i<TaskList.size();i++)
    {
        cudaSetDevice(dev_id);
        int tid = omp_get_thread_num();
        execute(TaskList[i], handles, streams, tid);
    }
    
    cudaDeviceSynchronize();

    for (int i=0; i<nthreads+1;i++)
    {
        cudaStreamDestroy(streams[i]);
        cublasDestroy(handles[i]);
    }
    
    std::vector< Task<double> >().swap(TaskList);


    cudaDeviceReset();
    return 0;
}
