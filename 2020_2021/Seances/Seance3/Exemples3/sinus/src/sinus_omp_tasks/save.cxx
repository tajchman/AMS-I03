#include "save.hxx"
#include <fstream>

void save(const char *filename,
	  std::vector<double> & pos,
	  std::vector<double> & v1,
	  std::vector<double> & v2)
{
  std::ofstream f(filename);

  f  << "# x sin(systeme) approximation" << std::endl;
  int i, n = pos.size();
  for (i=0; i<n; i++)
    f << pos[i] << " " << v1[i] << " " << v2[i] << std::endl;
  
  std::ofstream t("sinus.gnp");
  t << "set output 'sinus.pdf'\n"
    << "set term pdf\n"
    << "plot 'sinus.dat' using 1:2 notitle w l lw 3, 'sinus.dat' using ($1):($3+0.03) notitle w l lw 3";
  
  t << std::endl;
}

