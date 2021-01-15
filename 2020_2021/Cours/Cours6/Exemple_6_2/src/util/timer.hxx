#ifndef _TIMER_HXX
#define _TIMER_HXX

#include <string>
#include <vector>
#include <iostream>

#include <chrono>

class Timer {
public:
  Timer(const char * s = 0L) : m_elapsed(0.0), m_running(false) {
    m_name = s ? s : "";
  }
  
  inline void reinit() { m_elapsed = 0.0; m_running = false; }
 
  const std::string & name() const { return m_name; }
  std::string & name() { return m_name; }
  
  void start() {
    if (m_running == false) {
      m_running = true;
      m_start = std::chrono::high_resolution_clock::now();
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
      std::chrono::duration<double> diff 
         = std::chrono::high_resolution_clock::now() - m_start;

      m_elapsed += diff.count();
      m_running = false;
    }
  }
  
  inline double elapsed() const { return m_elapsed; }
  
protected:
  
  std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
  double m_elapsed;
  bool m_running;
  std::string m_name;
};

void AddTimer(const char *name);
Timer &  GetTimer(int n);

#endif
