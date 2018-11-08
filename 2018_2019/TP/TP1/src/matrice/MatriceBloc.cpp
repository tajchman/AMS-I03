/*
 * Matrice.cpp
 *
 *  Created on: 5 nov. 2018
 *      Author: marc
 */
#include "MatriceBloc.hpp"
#include <iomanip>

std::ostream & operator<<(std::ostream & f, const MatriceBloc & A)
{
	int i,j, k,l, p=A.p(), q=A.q(), n=A(0,0).n(), m=A(0,0).m();
	for (k=0; k<p; k++)
		for (i=0; i<n; i++) {
			for (l=0; l<q; l++)
				for (j=0; j<m; j++)
					f << std::setw(12) << A(k,l)(i,j);
		f << std::endl;
		}
	return f;
}

void init(MatriceBloc &A)
{
	int i,j;
	int n=A(0,0).n(), m=A(0,0).m(), p=A.p(), q=A.q(), i0, j0;
	for (i=0, i0=0; i<p; i++, i0+=n)
		for (j=0, j0=0; j<q; j++, j0+=m)
			init(A(i,j), i0, j0);
}

void transpose(MatriceBloc & B, const MatriceBloc & A)
{
	int i,j;
	int p=A.p(), q=A.q();

	for (i=0; i<p; i++)
		for (j=0; j<q; j++)
			transpose(B(j,i),A(i,j));
}
