#include <cmath>
#include "Heat.hxx"

double Solver::Difference() {
  
  double diff = 0.0;

  int i, j, n = m_u.n(), m = m_u.m();
  
  for (i=0; i<n; i++)
    for (j=0; j<m; j++) {
      diff += std::abs(m_v(i,j) - m_u(i,j));
    }

  return diff;
}
