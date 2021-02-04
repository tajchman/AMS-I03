/*
 * Timer.hxx
 *
 *  Created on: 5 janv. 2016
 *      Author: tajchman
 */

#include <string>

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
	void name(const char *s) { m_name=s; }
	const char * name() { return m_name.c_str(); }
private:
	std::string m_name;
	sTime m_t0;
	bool m_running;
	double m_elapsed;
};

#endif /* TIMER_HXX_ */
