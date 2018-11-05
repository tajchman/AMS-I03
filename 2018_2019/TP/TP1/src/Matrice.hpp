/*
 * Matrice.hpp
 *
 *  Created on: 5 nov. 2018
 *      Author: marc
 */

#ifndef MATRICE_HPP_
#define MATRICE_HPP_
#include <vector>

template <typename T>
class Matrice {
public:
	Matrice<T>(int n=0, int m=0) : m_n(n), m_m(m), m_coefs(n*m) {}

	void resize(int n, int m) { m_n = n; m_m = m; m_coefs.resize(m_n, m_m); }
	int n() const { return m_n; }
	int m() const { return m_m; }
	T operator()(int i,int j) const { return m_coefs[i*m_m + j]; }
	T & operator()(int i,int j) { return m_coefs[i*m_m + j]; }
protected:
	int m_n, m_m;
	std::vector<T> m_coefs;
};

inline
void transpose(const double & A, double & B)
{
	B = A;
}

template<typename T>
void transpose(const Matrice<T> & A, Matrice<T> & B)
{
	int n = A.n(), m = A.m(), i,j;

	for(i=0; i<n; i++)
		for(j=0; j<m; j++)
			transpose(A(i,j), B(j,i));
}

#endif /* MATRICE_HPP_ */
