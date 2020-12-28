/*
 * gpu_scheme.hxx
 *
 *  Created on: 4 févr. 2018
 *      Author: marc
 */

#ifndef GPUSCHEME_HXX_
#define GPUSCHEME_HXX_

#include "AbstractScheme.hxx"
#include "CpuValues.hxx"
#include "GpuValues.hxx"
#include "GpuParameters.hxx"

class GpuScheme : public AbstractScheme {
public:
  GpuScheme(const GpuParameters *);
  ~GpuScheme();
  
  void initialize();
  bool iteration();
  const AbstractValues & getOutput();
  void setInput(const AbstractValues & u);

protected:
  double * dev_out;
  double * host_out;
  size_t numOutputElements;

  CpuValues m_u2, m_v2;
};

#endif /* GPUSCHEME_HXX_ */

