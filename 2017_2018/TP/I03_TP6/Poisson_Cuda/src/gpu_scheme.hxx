/*
 * GPUScheme.hpp
 *
 *  Created on: 4 févr. 2018
 *      Author: marc
 */

#ifndef GPUSCHEME_HPP_
#define GPUSCHEME_HPP_

#include "scheme.hxx"

struct sGPU;

class GPUScheme : public Scheme {
public:
	GPUScheme();
	virtual ~GPUScheme();
    virtual bool iteration();

private:
	sGPU * m_GPU;
};

#endif /* GPUSCHEME_HPP_ */

