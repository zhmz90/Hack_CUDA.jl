#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <linux/version.h>

#include <julia.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX_DEVICES 2
#define DATA_BUF_SIZE 4096

typedef struct ipcCUDA_st
{
    int device;
    pid_t pid;
    cudaIpcEventHandle_t eventHandle;
    cudaIpcMemHandle_t memHandle;
} ipcCUDA_t;

typedef struct ipcDevices_st
{
    int count;
    int ordinals[MAX_DEVICES];
} ipcDevices_t;

typedef struct ipcBarrier_st
{
    int count;
    bool sense;
    bool allExit;
} ipcBarrier_t;

ipcBarrier_t *g_barrier = NULL;
bool g_procSense;
int g_processCount;

void procBarrier()
{
    int newCount = __sync_add_and_fetch(&g_barrier->count, 1); // ???

    if (newCount == g_processCount)
    {
        g_barrier->count = 0;
        g_barrier->sense = !g_procSense;
    }
    else
    {
        while (g_barrier->sense == g_procSense)
        {
            if (!g_barrier->allExit)
            {
                sched_yield(); // ???
            }
            else
            {
                exit(EXIT_FAILURE);
            }
        
        }
    }

    g_procSense = !g_procSense;
}


__global__ void add_num(int *dst, const int *src, const int num, const int len)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gtid < len)
    {
        dst[gtid] = src[gtid] + num;
    }
}

void getDeviceCount(ipcDevices_t *devices)
{
    pid_t pid = fork();

    if (0 == pid)
    {
        int i;
        int count, uvaCount = 0;
        int uvaOrdinals[MAX_DEVICES];
        cudaGetDeviceCount(&count);
    
        for (i=0;i<count;i++)
        {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            
            if (prop.unifiedAddressing)
            {
                uvaOrdinals[uvaCount] = i;
                printf("GPU is capble of UVA\n");
                uvaCount++;
            }
            if (prop.computeMode != cudaComputeModeDefault)
            {
                printf("device must be computeMode\n");
                exit(EXIT_SUCCESS);
            }
        }
        
        devices->ordinals[0] = uvaOrdinals[0];
        
        printf("checking p2p mem acess...\n");
        devices->count = 1;
        int canAccessPeer_0i, canAccessPeer_i0;

        for (i=1; i < uvaCount; i++)
        {
            cudaDeviceCanAccessPeer(&canAccessPeer_0i, uvaOrdinals[0], uvaOrdinals[i]);
            cudaDeviceCanAccessPeer(&canAccessPeer_0i, uvaOrdinals[i], uvaOrdinals[0]);
           
            if (canAccessPeer_0i*canAccessPeer_i0)
            {
                devices->ordinals[devices->count] = uvaOrdinals[i];
                printf("p2p between GPU %d and GPU %d:YES\n",
                        devices->ordinals[0], devices->ordinals[1]);

            }

        }

    }
    else
    {
        int status;
        waitpid(pid, &status, 0);
        assert(!status);
    }

}

inline bool IsAppBuiltAs64()
{
    return sizeof(void *) == 8;
}

void test_add_num(ipcCUDA_t *s_mem, int index)
{
    int *d_ptr;
    int h_refData[DATA_BUF_SIZE];

    for (int i=0; i < DATA_BUF_SIZE; i++)
    {   
        h_refData[i] = rand();
    }

    cudaSetDevice(s_mem[index].device);

    if (0 == index)
    {
        printf("\nLaunching kernels...\n");
        
        int h_results[DATA_BUF_SIZE*MAX_DEVICES * PROCESSES_PER_DEVICE];
    
        cudaEvent_t event[MAX_DEVICES * PROCESSES_PER_DEVICE];
        cudaMalloc((void **)&d_ptr, DATA_BUF_SIZE*g_processCount*sizeof(int));
        cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&s_mem[0].memHandle, (void *)d_ptr);
        cudaMemcpy((void *)d_ptr, (void *)h_refData DATA_BUF_SIZE*sizeof(int), cudaMemcpyHostToDevice);

        procBarrier();

        for (int i=1; i < g_processCount; i++)
        {
            cudaIpcOpenEventhandle(&event[i],s_mem[i].eventHandle);
        }
        
        procBarrier();
        
        for (int i=1; i<g_processCount; i++)
        {
            cudaEventSynchronize(event[i]);
        }

        procBarrier();

        for (int i=1; i<g_processCount; i++)
        {
            cudaEventSynchronize(event[i]);
        }
        
        procBarrier();

        cudaMemcpy(h_results, d_ptr+DATA_BUF_SIZE,DATA_BUF_SIZE*(g_processCount - 1)*sizeof(int),
                cudaMemcpyDeviceToHost);
        cudaFree(d_ptr);
    
       
        
    }
    else
    {
        cudaEvent_t event;
        cudaEventCreate(&event, cudaEventDisableTiming | cudaEventInterprocess);
        cudaIpcGetEventHandle((cudaIpcEventHandle_t *)&s_mem[index].eventHandle, event);
    
        procBarrier();

        cudaIpcOpenMemHandle((void **)&d_ptr, s_mem[0].memHandle, cudaIpcMemLazyEnablePeerAccess);

        printf("index:%d,s_mem[index].device:%d,%d,s_mem[0].device:%d\n",index,s_mem[index].device,0,s_mem[0].device);

        const dim3 threads(512, 1);
        const dim3 blocks(DATA_BUF_SIZE/threads.x, 1);
        add_num<<<blocks,threads>>>(d_ptr+index*DATA_BUF_SIZE, d_ptr, index+1);
        cudaEventRecord(event);

        procBarrier();

        cudaIpcCloseMemHandle(d_ptr);
        
        procBarrier();

        cudaEventDestroy(event);
    }

}



int main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = &argv;

    ipcDevices_t *s_devices = (ipcDevices_t *) mmap(NULL, sizeof(*s_devices),
            PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, 0, 0);

    assert(MAP_FAILED != s_devices); // MAP_FAILED ???

    getDeviceCount(s_devices);

    if (s_devices->count > 1)
    {
        g_processCount = PROCESS_PER_DEVICE * s_devices->count;
    }
    
    g_barrier = (ipcBarrier_t *)mmap(NULL, sizeof(*g_barrier),
            PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, 0, 0);
    assert(MAP_FAILED != g_barrier);
    memset((void *)g_barrier, 0, sizeof(*g_barrier));

    g_procSense = 0;

    ipcCUDA_t *s_mem = (ipcCUDA_t *)mmap(NULL, g_processCount*sizeof(*s_mem),
            PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, 0, 0);
    assert(MAP_FAILED != s_mem);

    memset((void *)s_mem, 0, g_processCount*sizeof(*s_mem));

    int index = 0;

    for (int i=1; i < g_processCount;i++)
    {
        int pid = fork();

        if (!pid)
        {
            index = i;
            break;
        }
        else
        {
            s_mem[i].pid = pid;
        }
    }

    if (s_devices->count > 1)
    {
        s_mem[index].device = s_devices->ordinals[index / PROCESSES_PER_DEVICE];
    
    }
    else
    {
        s_mem[0].device = s_mem[1].device = s_devices->ordinals[0];
    }


    test_add_num(s_mem, index);

    if (0 == index)
    {
        for (int i=1;i<g_processCount;i++)
        {
            int status;
            waitpid(s_mem[i].pid, &status, 0);
            assert(WIFEXITED(status));
        }
        
        for (int i=0;i<s_devices->count;i++)
        {
            cudaSetDevices(s_devices->ordinals[i]);
            cudaDeviceReset();
        }
        
        exit(EXIT_SUCCESS);
    }


}
