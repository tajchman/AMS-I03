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
  Scheme(const Parameters& P, callback_t f);
  ~Scheme();
  size_t getDomainSize(int dim) const;

  double present();

  bool iteration();
  double variation() { return m_duv; }

  void initialize();
  void terminate();

  const Values& getOutput();
  void setInput(Values&& u);

  std::string codeName;

protected:
  double iteration_domaine(int imin, int imax, 
                           int jmin, int jmax,
                           int kmin, int kmax);
  const Parameters& m_P;
  Values m_u, m_v;
  double m_duv;
  callback_t m_f;

  double m_t, m_dt;
  size_t m_n[3];
  double m_dx[3];
  double m_xmin[3];
};

#endif /* SCHEME_HXX_ */
