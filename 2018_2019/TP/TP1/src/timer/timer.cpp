/*
 * timer.cpp
 *
 *  Created on: 12 nov. 2018
 *      Author: marc
 */

#include <chrono>
#include "timer.hpp"

void Timer::start()
{
   m_start = std::chrono::high_resolution_clock::now();
}

void Timer::end()
{
   m_end = std::chrono::high_resolution_clock::now();
   m_elapsed = (m_end-m_start).count();
}

