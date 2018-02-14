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
#include <builtin_types.h>

class GpuValues : public AbstractValues {
public:
	GpuValues(const GpuParameters * p);
	virtual ~GpuValues() {}
	double * g_data() { return g_u; }
	const double * g_data() const { return g_u; }

    void init();
	void init_f();

protected :
	void allocate(size_t nn);
	void deallocate();
	double * g_u;
};

#endif /* GPUVALUES_HXX_ */
