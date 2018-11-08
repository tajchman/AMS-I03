/*
 * Matrice.hpp
 *
 *  Created on: 5 nov. 2018
 *      Author: marc
 */

#ifndef MATRICE_BLOC_HPP_
#define MATRICE_BLOC_HPP_
#include <vector>
#include <iostream>
#include "Matrice.hpp"

class MatriceBloc {
public:
	MatriceBloc(int n=0, int m=0, int bp=1, int bq=1) : m_p(n/bp), m_q(n/bq), m_coefs(m_p*m_q)
	{
		int i;
		for (i=0; i<m_p*m_q; i++) m_coefs[i].resize(bp,bq);
	}

	int p() const { return m_p; }
	int q() const { return m_q; }
	const Matrice & operator()(int i,int j) const { return m_coefs[i*m_q + j]; }
	Matrice & operator()(int i,int j) { return m_coefs[i*m_q + j]; }
protected:
	int m_p, m_q;
	std::vector<Matrice> m_coefs;
};

void init(MatriceBloc &A);
std::ostream & operator<<(std::ostream & f, const MatriceBloc & A);

void transpose(MatriceBloc & B, const MatriceBloc & A);

#endif /* MATRICE_HPP_ */
