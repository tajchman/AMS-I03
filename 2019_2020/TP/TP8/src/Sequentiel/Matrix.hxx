#ifndef __MATRIX_HXX__
#define __MATRIX_HXX__

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

class Matrix {

public:

  typedef double (*fct) (double, double);
  
  Matrix(int n, int m, const char * name = "")
    : m_n(n), m_m(m), m_coefs(n*m), m_name(name) {}

  Matrix(int n, int m, Matrix::fct f, const char * name = "");

  void operator= (const Matrix & other) {
    m_n = other.m_n;
    m_m = other.m_m;
    m_coefs = other.m_coefs;
  }
  
  double operator()(int i, int j) const  { return m_coefs[i*m_m + j]; }
  double & operator()(int i, int j)      { return m_coefs[i*m_m + j]; }

  double * operator[](int i)             { return m_coefs.data() + i*m_m; }
  const double * operator[](int i) const { return m_coefs.data() + i*m_m; }

  int n() const { return m_n; }
  int m() const { return m_m; }
  const char * name() const { return m_name.c_str(); }
  void name(const char *s) { m_name = s; }

  void save(int kSave) const;
  
  static void swap(Matrix &u, Matrix &v) {
    std::swap(u.m_coefs, v.m_coefs);
    std::swap(u.m_m, v.m_m);
    std::swap(u.m_n, v.m_n);
  }

private:
  std::vector<double> m_coefs;
  int m_m, m_n;
  std::string m_name;
};

inline std::ostream & operator<< (std::ostream &f, const Matrix & u) {

  int i,j;
  f << u.name() << std::endl;
  
  for (i=0; i<u.n(); i++) {
    for (j=0; j<u.m(); j++)
      f << std::setw(12) << u(i,j);
    f << std::endl;
  }
  f << std::endl;

  return f;
}

void save(int ksave, const Matrix & M);

#endif

