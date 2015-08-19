#include "multithreading.h"

// create thread
CUTThread cutStartThread(CUT_THREADROUTINE func, void *data)
{
    pthread_t thread;
    pthread_create(&thread, NULL,func,data);
    return thread;
}

// wait for thread to finish
void cutEndThread(CUTThread thread)
{
    pthread_join(thread, NULL);
}

// wait for multiple threads
void cutWaitForThreads(const CUTThread *threads, int num)
{
    for (int i=0; i<num; i++)
    {
        cutEndThread(threads[i]);
    }
}

// create barrier
CUTBarrier cutCreateBarrier(int releaseCount)
{
    CUTBarrier barrier;

    barrier.count = 0;
    barrier.releaseCount = releaseCount;

    pthread_mutex_init(&barrier.mutex,0);
    pthread_cond_init(&barrier.conditionVariable,0);

    return barrier;
}

/*
void cutIncrementBarrier(CUTBarrier *barrier)
{
    int myBarrierCount;
    EnterCriticalSection(&barrier->criticalSection);
    myBarrierCount = ++barrier->count;
    LeaveCriticalSection(&barrier->criticalSection);

    if (myBarrierCount >= barrier->releaseCount)
    {
        SetEvent(barrier->barrierEvent);
    }
}
*/
// Increment barrier
void cutIncrementBarrier(CUTBarrier *barrier)
{
    int myBarrierCount;
    pthread_mutex_lock(&barrier->mutex);
    myBarrierCount = ++barrier->count;
    pthread_mutex_unlock(&barrier->mutex);

    if (myBarrierCount >= barrier->releaseCount)
    {
        pthread_cond_signal(&barrier->conditionVariable);
    }
}


// Wait for barrier release
void cutWaitForBarrier(CUTBarrier *barrier)
{
    pthread_mutex_lock(&barrier->mutex);

    while (barrier->count < barrier -> releaseCount)
    {
        pthread_cond_wait(&barrier->conditionVariable, &barrier->mutex);
    }

    pthread_mutex_unlock(&barrier->mutex);
}

// Destroy barrier
void cutDestroyBarrier(CUTBarrier *barrier)
{
    pthread_mutex_destroy(&barrier->mutex);
    pthread_cond_destroy(&barrier->conditionVariable);

}
