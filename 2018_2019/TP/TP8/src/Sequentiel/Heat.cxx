#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include "Heat.hxx"
#include "Params.hxx"

Solver::Solver(const sParams & p)
  : m_u(p.n, p.n, "u_solver"), 
    m_v(p.n, p.n, "v_solver"), 
    m_f(p.n, p.n, "f_solver")
{
  m_dx = 1.0/(p.n-1);
  m_dt_max = m_dt = 0.5*m_dx*m_dx;
  m_lambda = 0.25*m_dt/(m_dx*m_dx);
  m_t = 0.0;
}

void Solver::setForce(const Matrix &f)
{
  m_f = f;
}

void Solver::setInput(const Matrix &u)
{
  m_u = u; m_v = m_u;
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
  m_lambda = 0.25*m_dt/(m_dx*m_dx);
}

void Solver::Shift()
{
  Matrix::swap(m_u,m_v);
}

