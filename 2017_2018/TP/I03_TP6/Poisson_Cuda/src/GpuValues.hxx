/*
 * GPUValues.hpp
 *
 *  Created on: 4 févr. 2018
 *      Author: marc
 */

#ifndef GPUVALUES_HPP_
#define GPUVALUES_HPP_

#include "AbstractValues.hxx"

class GpuValues : public AbstractValues {
public:
  GpuValues(const Parameters * p);
  virtual ~GpuValues() {}

protected :
    void allocate(size_t nn);
    void deallocate();
};

#endif /* GPUVALUES_HPP_ */
