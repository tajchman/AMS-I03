#ifndef __TIME_WINDOWS__
#define __TIME_WINDOWS__

#include <stdint.h>
#include <windows.h>

#ifdef __cplusplus
extern "C" {
#endif

	struct sTime {
		LARGE_INTEGER frequency;        // ticks per second
		LARGE_INTEGER t0;
	};

	void initTime(struct sTime *t);
	double elapsedTime(struct sTime *t);

#ifdef __cplusplus
}
#endif

#endif
