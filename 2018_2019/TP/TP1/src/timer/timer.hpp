#ifndef _TIMER_HPP
#define _TIMER_HPP

#include <chrono>

class Timer {
  public:
    Timer() : m_elapsed(0.0), m_running(false) {}
    inline void reinit() { m_elapsed = 0.0; m_running = false; }
    void start();
    void stop();
    inline double elapsed() { return m_elapsed; }
  protected:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start, m_end;
    double m_elapsed;
    bool m_running;
};

#endif
