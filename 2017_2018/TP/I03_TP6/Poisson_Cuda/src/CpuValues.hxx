#ifndef CPUVALUES_HXX_
#define CPUVALUES_HXX_

#include "AbstractValues.hxx"
#include "CpuParameters.hxx"
#include <vector>
#include <iostream>

class CpuValues : public AbstractValues {

public:

  CpuValues(const CpuParameters * p);
  virtual ~CpuValues() {}
  void operator= (const CpuValues &);
  void init();
  void init_f();

protected:
  
  CpuValues(const CpuValues &other) : AbstractValues(other) {};

  virtual void allocate(size_t nn);
  virtual void deallocate();
};
				   

#endif
