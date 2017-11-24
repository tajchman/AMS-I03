#ifdef _WIN32

#include "time_windows.h"

void initTime(struct sTime *t)
{
	QueryPerformanceFrequency(&(t->frequency));
	QueryPerformanceCounter(&(t->t0));
}

double elapsedTime(struct sTime *t)
{
	LARGE_INTEGER t1; 
	QueryPerformanceCounter(&t1);
	return ((double) (t1.QuadPart - t->t0.QuadPart)) / t->frequency.QuadPart;
}

#endif