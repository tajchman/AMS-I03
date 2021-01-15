#include "timerGPU.hxx"
#include <vector>

static std::vector<TimerGPU> Timers;

void AddTimer(const char *name)
{
  Timers.push_back(TimerGPU(name));
}

TimerGPU & GetTimer(int n)
{
  return Timers[n];
}