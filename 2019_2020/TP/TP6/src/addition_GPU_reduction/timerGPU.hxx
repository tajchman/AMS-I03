#ifndef _TIMERGPU_HXX
#define _TIMERGPU_HXX

#include <string>

class TimerGPU {
public:
  TimerGPU(const char * s = 0L) : m_elapsed(0.0), m_running(false) {
    m_name = s ? s : "";
    cudaEventCreate(&m_start);
    cudaEventCreate(&m_stop);
  }
  
  inline void reinit() { m_elapsed = 0.0; m_running = false; }
 
  std::string & name() { return m_name; }
  
  void start() {
    if (not m_running) {
      m_running = true;
      cudaEventRecord(&m_start);
    }
  }
  
  void stop() {
    if (m_running) {
      m_running = false;
      cudaEventRecord(&m_stop);
      cudaEventSynchronize(stop);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      m_elapsed += milliseconds;
    }
  }
  
  inline double elapsed() { return m_elapsed; }
  
protected:
  
  cudaEvent_t m_start, m_stop;
  double m_elapsed;
  bool m_running;
  std::string m_name;
};

#endif
