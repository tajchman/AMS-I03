#ifndef __TIMERCL__

#include "timer.hxx"
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.hpp>
#endif

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
