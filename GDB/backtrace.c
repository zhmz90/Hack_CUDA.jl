#include <stdio.h>
#include <stdlib.h>



void func_slave()
{
    char *name = "function slave";
    printf("%s\n", name);

}

void func_master()
{
    char *name = "function master";
    printf("%s\n", name);
    
    int *data;
    int num = 3;
    data = (int *)malloc(sizeof(int)*num);
    
    for (int i=0; i<num;i++)
    {
        data[i] = i;
    }

    free(data);
    func_slave();
}

int main()
{
    func_master();
    return 0;
}
