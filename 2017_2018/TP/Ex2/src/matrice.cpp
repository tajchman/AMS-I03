#include "matrice.hpp"

Matrice::Matrice(size_t n, size_t m)
  : m_n(n), m_m(m), m_v(new double[n*m]) {
}

Matrice::~Matrice() {
  delete [] m_v;
}

