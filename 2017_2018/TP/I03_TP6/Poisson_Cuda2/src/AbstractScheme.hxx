/*
 * AbstractScheme.hxx
 *
 *  Created on: 12 févr. 2018
 *      Author: marc
 */

#ifndef ABSTRACTSCHEME_HXX_
#define ABSTRACTSCHEME_HXX_

#include <vector>
#include "AbstractValues.hxx"
#include "AbstractParameters.hxx"
#include "timer.hxx"

class AbstractScheme {

public:

  AbstractScheme(const AbstractParameters *P);
  virtual ~AbstractScheme();
  size_t getDomainSize(int dim) const;

  virtual void initialize();
  double present();

  virtual bool iteration() = 0;
  bool solve(unsigned int nSteps);
  double variation();
  void terminate();
  virtual const AbstractValues & getOutput()  = 0;
  virtual void setInput(const AbstractValues & u) = 0;
  void save(const char * /*fName*/);
  Timer & timer(int k) { return m_timers[k]; }
  std::string codeName;
  std::string deviceName;

protected:
  double m_t, m_dt, m_lambda;
  size_t m_n[3];
  size_t m_dx[3];
  size_t m_di[3];

  AbstractValues *m_u, *m_v;
  double m_duv_max;
  const AbstractParameters *m_P;
  std::vector<Timer> m_timers;
  int kStep;
};

#endif /* ABSTRACTSCHEME_HXX_ */
