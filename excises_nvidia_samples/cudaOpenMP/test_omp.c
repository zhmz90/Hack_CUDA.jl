#include <stdio.h>
#include <omp.h>

int main(void)
{   
    omp_set_num_threads(10);
    #pragma omp parallel
    {
        printf("hello from thread %d\n",omp_get_thread_num());
    
    }



    return 0;
}
