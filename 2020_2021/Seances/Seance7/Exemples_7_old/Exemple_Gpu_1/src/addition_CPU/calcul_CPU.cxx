#include <cmath>
#include <iostream>
#include "calcul.hxx"
#include "timer.hxx"

Calcul_CPU::Calcul_CPU(int m)
{
  Timer & T = GetTimer(T_AllocId); T.start();
  
  n = m;
  h_u = new double[n];
  h_v = new double[n];
  h_w = new double[n];
  for(int i = 0; i < n; i++ ) {
    h_u[i] = 0.0;
    h_v[i] = 0.0;
    h_w[i] = 0.0;
  }
  
  T.stop();

 }

Calcul_CPU::~Calcul_CPU()
{
  Timer & T = GetTimer(T_FreeId); T.start();

  delete [] h_w;
  delete [] h_v;
  delete [] h_u;

  T.stop();
}


void Calcul_CPU::init()
{
  Timer & T = GetTimer(T_InitId); T.start();

  double x;

  for(int i = 0; i < n; i++ ) {
    x = double(i);
    h_u[i] = sin(x)*sin(x);
    h_v[i] = cos(x)*cos(x);
  }

  T.stop();
}

void Calcul_CPU::addition()
{
  Timer & T = GetTimer(T_AddId); T.start();
  
  for (int i=0; i<n; i++)
    h_w[i] = h_u[i] + h_v[i];
  
  T.stop();
}

double Calcul_CPU::verification()
{
  Timer & T = GetTimer(T_VerifId); T.start();
  
  double s = 0;
  for (int i=0; i<n; i++)
    s += h_w[i];
  s = s/n - 1.0;
  
  T.stop();

  return s;
}

