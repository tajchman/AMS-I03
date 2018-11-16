/*
 * MatriceBloc.hpp
 *
 */

#ifndef MATRICEBLOC_HPP_
#define MATRICEBLOC_HPP_

#include <vector>
#include <iostream>
#include <string>
#include "Matrice.hpp"

class MatriceBloc {
public:
	MatriceBloc(int n=0, int m=0, int p=1, int q=1, const char *name = "") : m_n(n), m_m(m), m_p(p), m_q(q), m_coefs(n*m), m_name(name){
	   for(auto &e : m_coefs)
		   e.resize(p, q);
	}

	int n() const { return m_n; }
	int m() const { return m_m; }
	int p() const { return m_p; }
	int q() const { return m_q; }
	inline const Matrice & operator()(int i,int j) const { return m_coefs[i*m_m + j]; }
	inline Matrice & operator()(int i,int j) { return m_coefs[i*m_m + j]; }

	const char * name() const { return m_name.c_str(); }
protected:
	int m_n, m_m, m_p, m_q;
	std::vector<Matrice> m_coefs;
	std::string m_name;
};

void transpose(MatriceBloc & B, const MatriceBloc & A);

template <typename F>
void init(MatriceBloc &A, F & f)
{
  int i,j, n = A.n(), m = A.m(), p = A.p(), q = A.q();
  for (i=0; i<n; i++) {
    i0 = i*p;
    for (j=0; j<m; j++) {
        j0 = j*q;
        for (k=0; k<p; k++)
          for (l=0; l<q; l++)
            A(i,j)(k,l) = f(i0+p,j0+q);
}

std::ostream & operator<<(std::ostream & f, const MatriceBloc & A);

#endif /* MATRICEBLOC_HPP_ */
