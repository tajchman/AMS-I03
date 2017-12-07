/*
 * Timer.hxx
 *
 *  Created on: 5 janv. 2016
 *      Author: tajchman
 */

#ifndef TIMER_HXX_
#define TIMER_HXX_

#if defined(_WIN32)
#include  "time_windows.h"
#elif defined(__unix)
#include  "time_unix.h"
#endif

class Timer {
public:
	Timer() : m_running(false), m_elapsed(0.0) {}
	void reset();
	void start();
	void stop();
	double elapsed();

private:
	sTime m_t0;
	bool m_running;
	double m_elapsed;
};

#endif /* TIMER_HXX_ */
