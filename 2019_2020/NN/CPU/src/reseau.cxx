/*
 * reseau.cxx
 *
 *  Created on: 7 mars 2019
 *      Author: marc
 */

#include "reseau.hxx"

reseau::reseau(const std::vector<size_t> & layers)
   : m_nlayers(layers.size()),
	 m_neuron_layer(m_nlayers),
	 m_bias(m_nlayers),
	 m_weight(m_nlayers-1)
{
   size_t i;
   for (i=0; i<m_nlayers; i++) {
	   m_neuron_layer[i].resize(layers[i]);
	   m_bias[i].resize(layers[i]);
   }
   for (i=0; i<m_weight.size(); i++) {
	   m_weight[i].resize(layers[i], layers[i+1]);
   }

}
