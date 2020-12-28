#ifndef _TIMERGPU_HXX
#define _TIMERGPU_HXX

#include <string>

struct _TimerGPU;

class TimerGPU {

public:
  
  TimerGPU(const char *s =0L);

  ~TimerGPU();

  void start();

  void stop();
  
  double elapsed();
  
protected:

  bool m_running;
  double m_elapsed;
  std::string m_name;
  _TimerGPU *m_t;
};
  

#endif
