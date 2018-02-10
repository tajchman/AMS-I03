/*
 * GPUValues.hpp
 *
 *  Created on: 4 févr. 2018
 *      Author: marc
 */

#ifndef GPUVALUES_HPP_
#define GPUVALUES_HPP_

#include "values.hxx"

class GPUValues : public Values {
public:

protected :
    void allocate(size_t nn);
    void deallocate();
};

#endif /* GPUVALUES_HPP_ */
