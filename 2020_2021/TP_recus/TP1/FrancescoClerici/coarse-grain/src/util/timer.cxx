#include "timer.hxx"

static std::vector<Timer> Timers;

void AddTimer(const char *name)
{
  Timers.push_back(Timer(name));
}

Timer & GetTimer(int n)
{
  return Timers[n];
}