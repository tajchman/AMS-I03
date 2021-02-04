#include <cmath>
#include <iostream>
#include "Calcul_OpenMP.hxx"
#include "timer.hxx"

Calcul_OpenMP::Calcul_OpenMP(int m)
{
  Timer & T = GetTimer(T_AllocId); T.start();
  
  n = m;
  h_u = new double[n];
  h_v = new double[n];
  h_w = new double[n];
  
  T.stop();

 }

Calcul_OpenMP::~Calcul_OpenMP()
{
  Timer & T = GetTimer(T_FreeId); T.start();

  delete [] h_w;
  delete [] h_v;
  delete [] h_u;

  T.stop();
}


void Calcul_OpenMP::init()
{
  Timer & T = GetTimer(T_InitId); T.start();

  double x;

#pragma omp parallel for private(x)
  for(int i = 0; i < n; i++ ) {
    x = double(i);
    h_u[i] = sin(x)*sin(x);
    h_v[i] = cos(x)*cos(x);
  }

  T.stop();
}

void Calcul_OpenMP::addition()
{
  Timer & T = GetTimer(T_AddId); T.start();
  
#pragma omp parallel for
  for (int i=0; i<n; i++)
    h_w[i] = h_u[i] + h_v[i];
  
  T.stop();
}

double Calcul_OpenMP::somme()
{
  Timer & T = GetTimer(T_SommeId); T.start();
  
  double s = 0;
#pragma omp parallel for reduction(+:s)
  for (int i=0; i<n; i++)
    s += h_w[i];
  
  T.stop();

  return s;
}

