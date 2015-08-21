
__global__ void kernel(const int *in, int *out, int a);
__global__ void kernel(const int2 *in, int *out, int a);
__global__ void kernel(const int *in1, const int *in2, int *out, int a);


int main()
{
    void (*func1)(const int *, int *,int);
    void (*func2)(const int2 *, int *, int);
    void (*func3)(const int*, const int *, int *, int)

    cudaFuncSetCacheConfig();
    cudaFuncGetAttributes();

}
