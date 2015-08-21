#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cstring>

#include <cuda_runtime.h>
#include <julia.h>

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction vec_add;





