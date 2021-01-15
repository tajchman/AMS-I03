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
 
  const std::string & name() const { return m_name; }
  std::string & name() { return m_name; }
  
  void start() {
    if (m_running == false) {
      m_running = true;
      cudaEventRecord(m_start);
    }
  }

  void restart() {
    if (m_running == false) {
      reinit();
      start();
    }
  }
  
  void stop() {
    if (m_running) {
      cudaEventRecord(m_stop);
      cudaEventSynchronize(m_stop);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, m_start, m_stop);
      m_elapsed += milliseconds;
      m_running = false;
    }
  }
  
  inline double elapsed() const { return m_elapsed; }
  
protected:
  
  cudaEvent_t m_start, m_stop;
  double m_elapsed;
  bool m_running;
  std::string m_name;
};

void AddTimer(const char *name);
TimerGPU &  GetTimer(int n);

#endif
