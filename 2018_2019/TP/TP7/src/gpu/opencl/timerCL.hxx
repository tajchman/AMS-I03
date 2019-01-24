#ifndef __TIMERCL__

#include "timer.hxx"
#include <CL/cl.h>

class TimerCL : public Timer {
public:

  TimerCL(cl_command_queue & q) : Timer(), m_q(q) {
  }

  ~TimerCL() {
  }

  void stop() {
    clFinish(m_q);
    Timer::stop();
  }
  
  cl_command_queue & m_q;
};

#endif
