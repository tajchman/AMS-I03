/*
 * Matrice.hpp
 *
 *  Created on: 5 nov. 2018
 *      Author: marc
 */

#ifndef MATRICE_HPP_
#define MATRICE_HPP_
#include <vector>
#include <iostream>
#include <string>

class Matrice {
public:
	Matrice(int n=0, int m=0, const char *name) : m_n(n), m_m(m), m_coefs(n*m), m_name(name){}

	void resize(int n, int m) { m_n = n; m_m = m; m_coefs.resize(m_n*m_m); }
	int n() const { return m_n; }
	int m() const { return m_m; }
	double operator()(int i,int j) const { return m_coefs[i*m_m + j]; }
	double & operator()(int i,int j) { return m_coefs[i*m_m + j]; }

	const double * line(int i) const { return &m_coefs[i*m_m]; }
	const char * name() { return m_name.c_str(); }
protected:
	int m_n, m_m;
	std::vector<double> m_coefs;
	std::string m_name;
};

void transpose(Matrice & B, const Matrice & A);

void init(Matrice &A, int i0 = 0, int j0 = 0);
std::ostream & operator<<(std::ostream & f, const Matrice & A);

#endif /* MATRICE_HPP_ */
