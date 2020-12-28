#include <cmath>
#include <iostream>
#include "pause.hxx"

static int imax;

void set_terms(int n)
{
  imax = n;
}

double sinus_taylor(double x)
{
  double y = x, x2 = x*x;
   int i, m;
   double coef = x;
   for (i=1; i<imax; i++) {
     m = 2*i*(2*i+1);
     coef *= -x2/m;
     y += coef;
     if (std::abs(coef) < 1e-12)
       break;
   }

   pause(2*i*i);
   return y;
}

double sinus_machine(double x)
{
  double y = sin(x);
  return y;
}

