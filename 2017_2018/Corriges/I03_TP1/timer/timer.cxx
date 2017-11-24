/*
 * Timer.cxx
 *
 *  Created on: 5 janv. 2016
 *      Author: tajchman
 */

#include "timer.hxx"
#include <cstdlib>

void Timer::reset()
{
	m_elapsed = 0.0;
	m_running = false;
}

void Timer::start()
{
	if (m_running == true) return;

	initTime(&m_t0);
	m_running = true;
}

void Timer::stop()
{
	if (m_running == false) return;

	m_elapsed += elapsedTime(&m_t0);
	m_running = false;
}

double Timer::elapsed()
{
	return  m_elapsed;

}
