#include "timer.hxx"

static std::vector<Timer> Timers;

void AddTimer(const char *name)
{
  Timers.push_back(Timer(name));
}

Timer & GetTimer(int n)
{
  if (n < 0)
     return Timers[Timers.size()+n];
  else
     return Timers[n];
}

#include <iomanip>
void PrintTimers(std::ostream &f)
{
  f << "\nTemps de calcul:\n\n";
  for (auto & t : Timers)
    f << std::setw(15) << t.name() << ": " 
      << std::setw(10) << std::setprecision(5) << t.elapsed() << " s" << std::endl;
}
