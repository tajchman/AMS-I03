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
	Matrice<T>(int n, int m) : m_n(n), m_m(m), m_coefs(n*m) {}
	T operator()(int i,int j) const { return m_coefs[i*m_m + j]; }
	T & operator()(int i,int j) { return m_coefs[i*m_m + j]; }
protected:
	int m_n, m_m;
	std::vector<T> m_coefs;
};

#endif /* MATRICE_HPP_ */
