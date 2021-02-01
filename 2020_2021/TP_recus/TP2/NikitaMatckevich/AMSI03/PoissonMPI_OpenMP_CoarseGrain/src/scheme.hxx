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

  double present();

  bool iteration();
  double variation() { return m_duv; }

  const Values & getOutput();
  void setInput(const Values & u);

  void synchronize();

  std::string codeName;

protected:
  double m_t, m_dt;
  double m_dx[3];
  double m_xmin[3];

  double iteration_domaine(int imin, int imax, 
                           int jmin, int jmax,
                           int kmin, int kmax);

  Values m_u, m_v;
  double m_duv = 0.0, m_duv_proclocal = 0.0;
  Parameters &m_P;
  callback_t m_f;
	std::vector<double> m_bufferIn{};
	std::vector<double> m_bufferOut{};
};

#endif /* SCHEME_HXX_ */
