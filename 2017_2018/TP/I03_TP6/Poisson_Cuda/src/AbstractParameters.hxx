/*
 * AbstractParameters.hxx
 *
 *  Created on: 13 févr. 2018
 *      Author: marc
 */

#ifndef ABSTRACTPARAMETERS_HXX_
#define ABSTRACTPARAMETERS_HXX_

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "GetPot.hxx"

class AbstractParameters : public GetPot {
public:

  AbstractParameters(int argc, char **argv);
  virtual ~AbstractParameters();
  std::ostream & out();
  void info();

  int n(int i) const { return m_n[i]; }
  double dx(int i) const { return m_dx[i]; }
  double xmin(int i) const { return m_xmin[i]; }

  int imin(int i) const { return m_imin[i]; }
  int imax(int i) const { return m_imax[i]; }
  int di(int i) const { return m_di[i]; }

  int itmax() const { return m_itmax; }
  double dt() const { return m_dt; }

  int freq() const { return m_freq; }
  std::string resultPath() const { return m_path; }
  bool help();

protected:
  std::ostream * m_out;

  std::string m_command;
  int m_n[3];
  double m_xmin[3], m_dx[3];
  int m_imin[3], m_imax[3], m_di[3];

  int m_itmax;
  double m_dt;

  int m_freq;

  std::string m_path;
  bool m_help;

};

std::ostream & operator<<(std::ostream &f, const AbstractParameters & p);

#endif /* ABSTRACTPARAMETERS_HXX_ */
