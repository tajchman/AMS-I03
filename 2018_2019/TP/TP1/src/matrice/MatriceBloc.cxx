/*
 * Matrice.cxx
 *
 *  Created on: 5 nov. 2018
 *      Author: marc
 */
#include "MatriceBloc.hxx"
#include <iomanip>

std::ostream & operator<<(std::ostream & f, const MatriceBloc & A)
{
  int i,j,k,l, n = A.n(), m = A.m(), p = A.p(), q = A.q();
  
  f << A.name() << std::endl;
  for (i=0; i<n; i++)
    for (k=0; k<p; k++) {
      for (j=0; j<m; j++)
        for (l=0; l<q; l++)
          f << std::setw(12) << A(i,j)(k,l);
      f << std::endl;
    }
  
  return f;
}

