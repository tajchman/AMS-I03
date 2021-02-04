#include "timer.hxx"
#include <iomanip>

static std::vector<Timer> Timers;

int AddTimer(const char *name)
{
  Timers.push_back(Timer(name));
  return Timers.size() - 1;
}

Timer & GetTimer(int n)
{
  if (n < 0)
     return Timers[Timers.size()+n];
  else
     return Timers[n];
}

void PrintTimer(std::ostream &f, const std::string &name, 
                double t, double t_total)
{
  f << std::setw(15) << name << ": " << std::setw(13);
  if (t > 0.0 && t < 1e-3)
      f << std::scientific; 
    else
      f << std::fixed;
    
  f << std::setprecision(3) << t << " s"
    << std::setw(13) << std::setprecision(2) << std::fixed
    << t * 100.0/t_total << " %" << std::endl;
}

void PrintTimers(std::ostream &f)
{
  double T_total = GetTimer(-1).elapsed();
  double T_other = T_total;

  f << "\nTemps de calcul:\n\n";
  for (int i=0; i<Timers.size(); i++) {
    Timer & t = Timers[i];
    if (t.name() == "total") {
      PrintTimer(f, "other", T_other, T_total); 
      f << "        _______________________________________\n";
    }
    else
       T_other -= t.elapsed();
    
    PrintTimer(f, t.name(), t.elapsed(), T_total);
  }
  f << std::endl;
}
