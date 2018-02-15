/*
 * GPUValues.hpp
 *
 *  Created on: 4 févr. 2018
 *      Author: marc
 */

#ifndef GPUVALUES_HXX_
#define GPUVALUES_HXX_

#include "GpuParameters.hxx"
#include "AbstractValues.hxx"

class GpuValues : public AbstractValues {
public:
  GpuValues(const GpuParameters * p);
  virtual ~GpuValues() {}
  
  void init();
  void init_f();
  void print(std::ostream &) const  { throw std::string("non implemented"); }
  void plot(const char * prefix, int order) const { throw std::string("non implemented"); }

protected :
  void allocate(size_t nn);
  void deallocate();
};

#endif /* GPUVALUES_HXX_ */
