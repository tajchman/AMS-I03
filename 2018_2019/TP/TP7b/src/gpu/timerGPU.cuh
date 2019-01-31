#ifndef _TIMERGPU_HXX
#define _TIMERGPU_HXX

#include <cuda.h>
#include <iostream>
#include <string>

class TimerGPU {

public:
  
  TimerGPU(const char *s =0L) : m_running(false), m_elapsed(0.0) {
    cudaEventCreate(&m_startEvent);
    cudaEventCreate(&m_stopEvent);
  }

  ~TimerGPU() {
    cudaEventDestroy(m_startEvent);
    cudaEventDestroy(m_stopEvent);
  }

  void start() {
    if (not m_running) {
      cudaEventRecord(m_startEvent,0);
      m_running = true;
    }
  }

  void stop() {
    if (m_running) {
      float ms;

      cudaEventRecord(m_stopEvent,0);
      cudaEventSynchronize(m_stopEvent);
      cudaEventElapsedTime(&ms, m_startEvent, m_stopEvent);

      m_elapsed += ms;
      m_running = false;
    }
  }
  
  inline double elapsed() { return m_elapsed; }
  
protected:

  cudaEvent_t m_startEvent, m_stopEvent;
  bool m_running;
  double m_elapsed;
  std::string m_name;

};
  

#endif
