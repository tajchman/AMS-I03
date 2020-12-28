/*
 * scheme.hxx
 *
 *  Created on: 5 janv. 2016
 *      Author: tajchman
 */

#ifndef SCHEME_HXX_
#define SCHEME_HXX_

#include <vector>
#include "values.hxx"
#include "parameters.hxx"

class Scheme {

public:
  Scheme(Parameters &P, callback_t f);
  ~Scheme();
  size_t getDomainSize(int dim) const;

  double present();

  bool iteration();
  double variation() { return m_duv; }

  const Values & getOutput();
  void setInput(const Values & u);

  void synchronize();

  std::string codeName;

protected:
  double m_t, m_dt;
  size_t m_n[3];
  double m_dx[3];
  double m_xmin[3];

  double iteration_domaine(int imin, int imax, 
                           int jmin, int jmax,
                           int kmin, int kmax);

  Values m_u, m_v;
  double m_duv;
  Parameters &m_P;
  callback_t m_f;
};

#endif /* SCHEME_HXX_ */
