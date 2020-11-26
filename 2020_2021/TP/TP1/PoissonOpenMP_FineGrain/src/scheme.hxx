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
#include "timer.hxx"

class Scheme {

public:
  Scheme(const Parameters &P);
  ~Scheme();
  size_t getDomainSize(int dim) const;

  void initialize(callback_t f);
  double present();

  bool iteration();
  bool solve(unsigned int nSteps);
  double variation();
  void terminate();
  const Values & getOutput();
  void setInput(const Values & u);
  void save(const char * /*fName*/);

  Timer & timer(int k) { return m_timers[k]; }
  int ntimers() { return m_timers.size();}
  std::string codeName;

protected:
  double m_t, m_dt;
  size_t m_n[3];
  double m_dx[3];
  size_t m_di[3];
  double m_xmin[3];

  double iteration_domaine(int imin, int imax, 
                           int jmin, int jmax,
                           int kmin, int kmax);

  Values m_u, m_v;
  double m_duv;
  const Parameters &m_P;
  callback_t m_f;

  std::vector<Timer> m_timers;
  int kStep;
};

#endif /* SCHEME_HXX_ */
