/*
 * Matrice.cxx
 *
 *  Created on: 5 nov. 2018
 *      Author: marc tajchman
 */
#include "Matrice.hxx"
#include <iomanip>

std::ostream & operator<<(std::ostream & f, const Matrice & A)
{
  int i,j, n = A.n(), m = A.m();

  f << A.name() << std::endl;
  for (i=0; i<n; i++) {
    for (j=0; j<m; j++)
      f << std::setw(12) << A(i,j);
    f << std::endl;
  }
  return f;
}


void transpose(Matrice & B, const Matrice & A)
{
  int i,j, n = A.n(), m = A.m();

  for (i=0; i<n; i++)
    for (j=0; j<m; j++)
      B(j,i) = A(i,j);

}
