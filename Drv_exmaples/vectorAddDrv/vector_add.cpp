#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cstring>

#include <cuda.h>
#include <builtin_types.h>
#include <drvapi_error_string.h>

using namespace std;


CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction vec_add_kernel;

float *h_A;
float *h_B;
float *h_C;

CUdeviceptr d_A;
CUdeviceptr d_B;
CUdeviceptr d_C;
bool nopromp = false;

void Cleanup(bool);
CUresult CleanupNoFailure();
void RandomInit(float *, int);
bool findModulePath(const char *, string &, char **, string &);

int *pArgc = NULL;
char **pArgv = NULL;

#define PTX_FILE "vectorAdd_kernel.ptx"

inline int cudaDeviceInit(int ARGC, char **ARGV)
{
    int cuDevice = 0;
    int deviceCount = 0;
    Curesult err = cuInit(0);
    
    cuDeviceGetCount(&deviceCount);



}


int main()
{
    CUresult error;
    
    int N = 5000, devID=0;
    size_t size = N*sizeof(float);
    
    


    return 0;
}
