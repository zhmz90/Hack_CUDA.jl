#include <stdio.h>
#include <stdlib.h>

#include <julia.h>

int main()
{
    jl_init("/home/guo/julia/usr/lib");


    jl_eval_string("println(\"hello\")");


    /*
    jl_value_t* array_type = jl_apply_array_type(jl_float64_type,1);
    jl_array_t* x = jl_alloc_array_1d(array_type,10);

    JL_GC_PUSH1(&x);

    double* xData = jl_array_data(x);
    
    int i;
    for (i=0;i<10;i++)
    {
        xData[i] = i;
    }

    */
    JL_GC_POP();


    jl_atexit_hook(0);
    return 0;
}

