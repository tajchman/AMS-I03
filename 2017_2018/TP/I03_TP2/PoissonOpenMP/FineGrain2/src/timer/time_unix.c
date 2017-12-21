#ifdef __unix

#include <stdlib.h>
#include "time_unix.h"

void initTime(struct sTime *t0)
{
	gettimeofday(t0, NULL);
}

double elapsedTime(struct sTime *t0)
{
	struct sTime t1;
	double dt;
	gettimeofday(&t1, NULL);

	dt = t1.tv_sec - t0->tv_sec + 1e-6*(t1.tv_usec - t0->tv_usec);
	return dt;
}


#endif
