/*
 * timer.cpp
 *
 */

#include <chrono>
#include "timer.hpp"

void Timer::start()
{
  if (not m_running) {
    m_start = std::chrono::high_resolution_clock::now();
    m_running = true;
  }
}

void Timer::stop()
{
  if  (m_running) {
    m_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = m_end-m_start;
    m_elapsed += diff.count();
    m_running = false;
  }
}

