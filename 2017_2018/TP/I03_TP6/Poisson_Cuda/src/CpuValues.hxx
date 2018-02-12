#ifndef __VALUES__
#define __VALUES__

#include "AbstractValues.hxx"
#include "parameters.hxx"
#include <vector>
#include <iostream>

class CpuValues : public AbstractValues {

public:

  CpuValues(const Parameters * p);
  virtual ~CpuValues() {}
  void operator= (const CpuValues &);

protected:
  
  CpuValues(const CpuValues &other) : AbstractValues(other) {};

  virtual void allocate(size_t nn);
  virtual void deallocate();
};
				   

#endif
