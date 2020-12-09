#include <cmath>
#include "calcul.hxx"
#include "sin.hxx"

void calcul(std::vector<double> & pos,
            std::vector<double> & v1,
            std::vector<double> & v2,
            int n1, int n2)
{
  double x;
  int i, n = pos.size();

  for (i=n1; i<n2; i++) {
    x = i*2*M_PI/n;
    pos[i] = x;
    v1[i] = sinus_machine(x);
    v2[i] = sinus_taylor(x);
  }
}
