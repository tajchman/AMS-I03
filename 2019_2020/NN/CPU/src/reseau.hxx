/*
 * reseau.hxx
 *
 *  Created on: 7 mars 2019
 *      Author: marc
 */

#ifndef RESEAU_HXX_
#define RESEAU_HXX_

#include "vector.hxx"
#include "matrix.hxx"
#include <math.h>

class reseau {
public:
	reseau(const std::vector<size_t> & layers);
	int forward(const std::vector<unsigned char> &v);

	double sigmoid(double x) { return 1.0/(1.0 + exp(x));}

private:
	size_t m_nlayers;
	std::vector<vector> m_neuron_layer;
	std::vector<vector> m_bias;
	std::vector<matrix> m_weight;

};

#endif /* RESEAU_HXX_ */
