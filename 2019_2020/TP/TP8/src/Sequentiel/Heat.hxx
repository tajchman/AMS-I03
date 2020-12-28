#ifndef __HEAT_XX__
#define __HEAT_XX__

#include "Matrix.hxx"

class sParams;

class Solver {
public:
  Solver(const sParams & p); 

  void setForce(const Matrix &f);
  
  void setInput(const Matrix &u);
  const Matrix & getOutput() const;

  void setTimeStep(double & dT);
  double getTime() const { return m_t; }
  
  void Iteration();
  double Difference();

  void Shift();
  
protected:
  Matrix m_u, m_v, m_f;
  double m_dt_max, m_dt, m_dx, m_lambda;
  double m_t;
};

#endif
