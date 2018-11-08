/*
 * Matrice.cpp
 *
 *  Created on: 5 nov. 2018
 *      Author: marc
 */
#include "Matrice.hpp"
#include <iomanip>

std::ostream & operator<<(std::ostream & f, const Matrice & A)
{
	int i,j, n = A.n(), m = A.m();
	for (i=0; i<n; i++) {
		for (j=0; j<m; j++)
			f << std::setw(12) << A(i,j);
		f << std::endl;
	}
	return f;
}

void init(Matrice &A, int i0, int j0)
{
	int i,j, n = A.n(), m = A.m();
	for (i=0; i<n; i++)
		for (j=0; j<m; j++)
			A(i,j) = 1.0/(i+i0+2*(j+j0)+1.0);
}

void transpose(Matrice & B, const Matrice & A)
{
	int i,j, n = A.n(), m = A.m();

	for (i=0; i<n; i++) {
	//	const double *p = A.line(i);
		for (j=0; j<m; j++)
			B(j,i) = A(i,j);
		//	B(j,i) = *p++;
}
}
