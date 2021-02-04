#include <cmath>
#include <iostream>
#include "Calcul_MPI.hxx"
#include "timer.hxx"

Calcul_MPI::Calcul_MPI(int m0, int m1)
{
  Timer & T = GetTimer(T_AllocId); T.start();
  
  n0 = m0; n1 = m1;
  n_local = n1-n0;
  h_u = new double[n_local];
  h_v = new double[n_local];
  h_w = new double[n_local];
  
  T.stop();

 }

Calcul_MPI::~Calcul_MPI()
{
  Timer & T = GetTimer(T_FreeId); T.start();

  delete [] h_w;
  delete [] h_v;
  delete [] h_u;

  T.stop();
}


void Calcul_MPI::init()
{
  Timer & T = GetTimer(T_InitId); T.start();

  double x;

  for(int i = 0; i < n_local; i++ ) {
    x = double(n0 + i);
    h_u[i] = sin(x)*sin(x);
    h_v[i] = cos(x)*cos(x);
  }

  T.stop();
}

void Calcul_MPI::addition()
{
  Timer & T = GetTimer(T_AddId); T.start();
  
  for (int i=0; i<n_local; i++)
    h_w[i] = h_u[i] + h_v[i];
  
  T.stop();
}

double Calcul_MPI::somme()
{
  Timer & T = GetTimer(T_SommeId); T.start();
  
  double s = 0;
  for (int i=0; i<n_local; i++)
    s += h_w[i];
  
  T.stop();

  return s;
}

