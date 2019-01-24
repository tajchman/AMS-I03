#ifndef __TIMERCL__

#include "timer.hxx"

class TimerCL public Timer {
public:

  TimerCL(cl_queue & q) : Timer(), m_q(q) {
  }

  ~TimerCL() {
  }

  void stop() {
    clFinish(m_q);
    Timer::stop();
  }
  
  cl_queue & m_q;
};

#endif
