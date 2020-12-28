#include "calcul.hxx"

void calcul1(std::vector<double> &y, 
             double a, 
             const std::vector<double> &x,
             double b) 
{
    int i, n = x.size();

    for (i=0; i<n; i++) {
      y[i] = a * x[i]+ b;
    }
}

void calcul2(std::vector<double> &y, 
             double a, 
             const std::vector<double> &x,
             double b) 
{
    int i, n = x.size();
    int i1, i2, i3;
    
    int N = n/4, m = N*4;
    for (i=0; i<m; i+=4) {
      y[i+0] = a * x[i+0]+ b;
      y[i+1] = a * x[i+1]+ b;
      y[i+2] = a * x[i+2]+ b;
      y[i+3] = a * x[i+3]+ b;
    }
    for (i=m; i<n; i++)
      y[i] = a * x[i] + b;
}

