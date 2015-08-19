#include <cstdio>
#include <cstdlib>
#include <iostream>

void f(int &i)
{
    i++;
}


int main(void)
{
    int i=5;
//    f(i);
    i = i/2;

    printf("i:%d\n",i);


    return 0;
}
