#include "calcul.hxx"
#include <cmath>
#include <fstream>

void calcul1(Matrice &C, const Matrice &A, const Matrice &B)
{
  int i,j,n = C.n(), m = C.m();

  for (i=0; i<n; i++)
    for (j=0; j<m; j++)
       C(i,j) = A(i,j) + B(i,j);
}

void calcul2(Matrice &C, const Matrice &A, const Matrice &B)
{
  int i,j,n = C.n(), m = C.m();

  for (j=0; j<m; j++)
    for (i=0; i<n; i++)
       C(i,j) = A(i,j) + B(i,j);
}
