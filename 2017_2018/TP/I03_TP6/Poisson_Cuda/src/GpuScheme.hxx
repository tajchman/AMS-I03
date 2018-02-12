/*
 * gpu_scheme.hxx
 *
 *  Created on: 4 févr. 2018
 *      Author: marc
 */

#ifndef GPUSCHEME_HXX_
#define GPUSCHEME_HXX_

#include "AbstractScheme.hxx"
#include "GpuValues.hxx"

struct sGPU;

class GpuScheme : public AbstractScheme {
public:
  GpuScheme(const Parameters *);
  ~GpuScheme();
  
  void initialize();
  bool iteration();

protected:
  sGPU * m_GPU;
  GpuValues m_duv, m_duv2;
};

#endif /* GPUSCHEME_HXX_ */

