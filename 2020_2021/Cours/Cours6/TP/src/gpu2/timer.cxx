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
  double T_total = GetTimer(-1).elapsed();

  f << "\nTemps de calcul:\n\n";
  for (auto & t : Timers) {
    f << std::setw(15) << t.name() << ": " << std::setw(13);
    if (t.elapsed() < 1e-2)
      f << std::scientific; 
    else
      f << std::fixed;
    
    f << std::setprecision(3) << t.elapsed() << " s"
      << std::setw(13) << std::setprecision(2) << std::fixed
      << t.elapsed() * 100.0/T_total << " %" << std::endl;
  }
  f << std::endl;
}
