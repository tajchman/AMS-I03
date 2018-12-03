#ifndef _TIMER_HXX
#define _TIMER_HXX

#ifdef _OPENMP
#include <omp.h>
#elif __cpluplus <= 199711L
#include <sys/time.h>
#else
#include <chrono>
#endif

class Timer {
  public:
    Timer() : m_elapsed(0.0), m_running(false) {}
  
    inline void reinit() { m_elapsed = 0.0; m_running = false; }
  
  void start() {
    if (not m_running) {
#ifdef _OPENMP
      m_start = omp_get_wtime();
#elif __cpluplus <= 199711L
      gettimeofday(&m_start, NULL);
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
#elif __cplusplus <= 199711L
      gettimeofday(&m_end  , NULL);
      m_elapsed += (m_end.tv_sec - m_start.tv_sec) 
                + 1e-6 * (m_end.tv_usec - m_start.tv_usec);
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
#elif __cplusplus <= 199711L
    struct timeval m_start, m_end;
#else
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start, m_end;
#endif
    double m_elapsed;
    bool m_running;
};

#endif
