#include "Heat.hxx"

void Solver::Iteration() {
  
  int i, j, n = m_u.n(), m = m_u.m();
  
  for (i=1; i<n-1; i++)
    for (j=1; j<n-1; j++) {
      m_v(i,j) = m_u(i,j)
	- m_lambda * (4*m_u(i,j)
                      - m_u(i+1,j) - m_u(i-1,j)
                      - m_u(i,j+1) - m_u(i,j-1))
        + m_f(i,j) * m_dt;
    }

  m_t += m_dt;
}

