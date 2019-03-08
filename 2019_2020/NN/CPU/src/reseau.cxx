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
	   m_weight[i].resize(layers[i+1], layers[i]);
   }

}

int reseau::forward(const std::vector<unsigned char> & image)
{
	size_t i,j,k,imax,kmax;
	double s;

	for (i=0; i<image.size(); i++)
		m_neuron_layer[0][i] = image[i]/255.0;

	for (j=0; j<m_nlayers-1; j++) {
		imax = m_neuron_layer[j].size();
	    kmax = m_neuron_layer[j+1].size();
	    for (k=0; k<kmax; k++) {
			s = m_bias[j][k];
	    	for(i=0; i<imax; i++)
	    		s += m_weight[j](k,i) * m_neuron_layer[j][i];
	    	m_neuron_layer[j+1][k] = sigmoid(s);
	    }
	}

	int idx = 0;
	double smax = m_neuron_layer[m_nlayers-1][0];
	for (i=1; i<m_neuron_layer[m_nlayers-1].size(); i++) {
		if (fabs(m_neuron_layer[m_nlayers-1][i]) > fabs(smax)) {
			idx = i;
			smax = m_neuron_layer[m_nlayers-1][i];
		}
	}
	return idx;
}
