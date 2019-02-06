#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include "Heat.hxx"
#include "Params.hxx"

Solver::Solver(const sParams & p)
  : m_u(p.n, p.n), m_v(p.n, p.n), m_f(p.n, p.n)
{
  m_dx = 1.0/(p.n-1);
  m_dt_max = m_dt = 0.5*m_dx*m_dx;
  m_lambda = m_dt/(m_dx*m_dx);
}

void Solver::setForce(const Matrix &f)
{
  m_f = f;
}

void Solver::setInput(const Matrix &u)
{
  m_u = u;
}

const Matrix & Solver::getOutput() const
{
  return m_v;
}

void Solver::setTimeStep(double & dT)
{
  if (dT > m_dt_max)
    dT = m_dt_max;

  m_dt = dT;
  m_lambda = m_dt/(m_dx*m_dx);
}

void Solver::Shift()
{
  Matrix::swap(m_u,m_v);
}

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
}

double Solver::Difference() {
  
  int i, j, n = m_u.n(), m = m_u.m();
  double diff = 0.0;
  
  for (i=0; i<n; i++)
    for (j=0; j<m; j++) {
      diff += std::abs(m_v(i,j) - m_u(i,j));
    }

  return diff;
}
