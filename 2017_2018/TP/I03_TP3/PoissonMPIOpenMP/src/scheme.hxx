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
  Scheme(const Parameters *P);
  ~Scheme();
  size_t getDomainSize(int dim) const;

  double present();
  bool solve(unsigned int nSteps);
  double variation();
  void terminate();
  const Values & getOutput();
  void setInput(const Values & u);
  void save(const char * /*fName*/);
  Timer & timer(int k) { return m_timers[k]; }
  std::string codeName;

protected:
  double m_t;
  size_t m_n[3];
  size_t m_dx[3];
  size_t m_di[3];

  Values m_u, m_v;
  double m_duv;
  const Parameters *m_P;
  std::vector<Timer> m_timers;
  int kStep;
};

#endif /* SCHEME_HXX_ */
