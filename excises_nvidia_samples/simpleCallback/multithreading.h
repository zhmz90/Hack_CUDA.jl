#include <pthread.h>

typedef pthread_t CUTThread;
typedef void *(*CUT_THREADROUTINE)(void *);

#define CUT_THREADPROC void*
#define CUT_THREADEND return 0

struct CUTBarrier
{
    pthread_mutex_t mutex;
    pthread_cond_t conditionVariable;
    int releaseCount;
    int count;
};


extern "C" {

CUTThread cutStartThread(CUT_THREADROUTINE, void *data);

void cutEndThread(CUTThread thread);

void cutWaitForThreads(const CUTThread *threads, int num);

CUTBarrier cutCreateBarrier(int releaseCount);

void cutIncrementBarrier(CUTBarrier *barrier);

void cutWaitForBarrier(CUTBarrier *barrier);

void cutDestroyBarrier(CUTBarrier *barrier);

}
