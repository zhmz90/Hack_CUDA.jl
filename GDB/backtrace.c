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
    
    func_slave();
}

int main()
{
    func_master();
    return 0;
}
