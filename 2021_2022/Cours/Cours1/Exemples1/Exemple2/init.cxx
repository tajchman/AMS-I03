#include "init.hxx"
#include <cstdlib>
#include <ctime>

void init(Matrice &M, double v0)
{
  int i,j,n = M.n(), m = M.m();

  for (i=0; i<n; i++)
    for (j=0; j<m; j++)
      M(i,j) = v0;
}
