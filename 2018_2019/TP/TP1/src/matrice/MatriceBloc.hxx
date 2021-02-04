/*
 * MatriceBloc.hxx
 *
 */

#ifndef MATRICEBLOC_HXX_
#define MATRICEBLOC_HXX_

#include <vector>
#include <iostream>
#include <string>
#include "Matrice.hxx"

class MatriceBloc {
public:
	MatriceBloc(int n=0, int m=0, int p=1, int q=1, const char *name = "") : m_n(n), m_m(m), m_p(p), m_q(q), m_coefs(n*m), m_name(name){
	for (int i=0; i<n*m; i++)
		   m_coefs[i].resize(p, q);
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
  std::vector<Matrice*> p_coefs;
  std::string m_name;
};

std::ostream & operator<<(std::ostream & f, const MatriceBloc & A);

#endif /* MATRICEBLOC_HXX_ */
