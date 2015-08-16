#include <stdio.h>
#include <stdlib.h>
#include <julia.h>

int main()
{
    jl_init("/home/guo/julia/usr/lib");
    
    jl_eval_string("print(sqrt(2.0))");

    jl_atexit_hook(0);

    return 0;
}
