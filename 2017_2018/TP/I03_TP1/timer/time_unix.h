
#ifndef __TIME_UNIX__
#define __TIME_UNIX__

#ifdef __unix

#include <stdint.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define sTime timeval

	void initTime(struct sTime *t);
	double elapsedTime(struct sTime *t);

#ifdef __cplusplus
}
#endif

#endif


#endif
