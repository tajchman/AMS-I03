/*
 * Matrice.hxx
 *
 */

#ifndef MATRICE_HXX_
#define MATRICE_HXX_

#include <vector>
#include <iostream>
#include <string>

class Matrice {
public:
	Matrice(int n=0, int m=0, const char *name = "")
          : m_n(n), m_m(m), m_coefs(n*m), m_name(name){}

	void resize(int n, int m) { m_n = n; m_m = m; m_coefs.resize(m_n*m_m); }
  int n() const { return m_n; }
  int m() const { return m_m; }
	inline double operator()(int i,int j) const { return m_coefs[i*m_m + j]; }
	inline double & operator()(int i,int j) { return m_coefs[i*m_m + j]; }

  const char * name() const { return m_name.c_str(); }

protected:
  int m_n, m_m;
  std::vector<double> m_coefs;
  std::string m_name;
};

std::ostream & operator<<(std::ostream & f, const Matrice & A);

#endif /* MATRICE_HXX_ */
