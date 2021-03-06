#ifndef _TIMER_HXX
#define _TIMER_HXX

#ifdef _OPENMP
#include <omp.h>
#else
#include <chrono>
#endif

class Timer {
  public:
    Timer() : m_elapsed(0.0), m_running(false) {}
  
    inline void reinit() { m_elapsed = 0.0; m_running = false; }
  
  void start() {
    if (!m_running) {
#ifdef _OPENMP
      m_start = omp_get_wtime();
#else
      m_start = std::chrono::high_resolution_clock::now();
#endif
      m_running = true;
    }
  }
  
  void stop() {
    if (m_running) {
#ifdef _OPENMP
      m_end = omp_get_wtime();
      m_elapsed += m_end - m_start;
#else
      m_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = m_end-m_start;
      m_elapsed += diff.count();
#endif
      m_running = false;
    }
  }
  
    inline double elapsed() { return m_elapsed; }
  protected:
  
#ifdef _OPENMP
  double m_start, m_end;
#else
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start, m_end;
#endif
    double m_elapsed;
    bool m_running;
};

#endif
