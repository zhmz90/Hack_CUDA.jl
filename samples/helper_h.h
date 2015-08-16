#define check_error(call) \
    if ((call) != cudaSuccess) \
    { \
        cudaError_t err = cudaGetLastError(); \
        printf("CUDA error calling \" "#call" \" \n"); \
    }

