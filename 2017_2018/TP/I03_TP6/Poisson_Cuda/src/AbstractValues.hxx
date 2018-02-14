/*
 * AbstractValues.hxx
 *
 *  Created on: 12 févr. 2018
 *      Author: marc
 */

#ifndef ABSTRACTVALUES_HXX_
#define ABSTRACTVALUES_HXX_

#include "AbstractParameters.hxx"

class AbstractValues {
public:

  AbstractValues(const AbstractParameters * prm);
  virtual ~AbstractValues() {}
  
  double * data() { return m_u; }
  const double * data() const { return m_u; }
  
  void swap(AbstractValues & other);
  
  virtual void init() = 0;
  virtual void init_f() = 0;
  
  virtual void print(std::ostream & f) const = 0;
  virtual void plot(const char * prefix, int order) const = 0;
  
  std::string codeName;
  std::string deviceName;
  
protected:
  virtual void allocate(size_t n) = 0;
  virtual void deallocate() = 0;
  
  void operator=(const AbstractValues & other);
  int n1, n2, nn;
  int m_n[3];
  double * m_u;
  const AbstractParameters * m_p;
  
};

#endif /* ABSTRACTVALUES_HXX_ */
