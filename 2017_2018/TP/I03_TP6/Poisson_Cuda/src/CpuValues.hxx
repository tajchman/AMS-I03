#ifndef CPUVALUES_HXX_
#define CPUVALUES_HXX_

#include "AbstractValues.hxx"
#include "CpuParameters.hxx"
#include <vector>
#include <iostream>

class CpuValues : public AbstractValues {

public:

  CpuValues(const CpuParameters * p);
  CpuValues(const CpuValues &other);
  virtual ~CpuValues() {}
  void operator= (const CpuValues &);
  void init();
  void init_f();

  double & operator() (int i,int j,int k) {
    return m_u[n2*i + n1*j + k];
  }
  double operator() (int i,int j,int k) const {
    return m_u[n2*i + n1*j + k];
  }
  
  void print(std::ostream &) const ;
  void plot(const char * prefix, int order) const ;

protected:
  
  virtual void allocate(size_t nn);
  virtual void deallocate();
};
				   

#endif
