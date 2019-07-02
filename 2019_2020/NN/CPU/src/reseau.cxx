/*
 * reseau.cxx
 *
 *  Created on: 7 mars 2019
 *      Author: marc
 */

#include "reseau.hxx"
#include <random>
#include <iostream>

reseau::reseau(const std::vector<size_t> & layers)
: m_nlayers(layers.size()),
  m_neuron_layer(m_nlayers),
  m_bias(m_nlayers),
  m_weight(m_nlayers-1)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  size_t i,j,k;

  for (i=0; i<m_nlayers; i++) {

//      std::cerr << "bias " << i << std::endl;

      size_t n = layers[i];
      m_neuron_layer[i].resize(n);
      m_bias[i].resize(n);

      for (j=0; j<n; j++)
	m_bias[i][j] = dis(gen);
//      for (j=0; j<n; j++)
//	std::cerr << "\t" << m_bias[i][j] << std::endl;
  }

  for (i=0; i<m_weight.size(); i++) {
      size_t n = layers[i+1], m = layers[i];
      std::cerr << "weight " << i << " (" << n << ", " << m << ")" << std::endl;

      m_weight[i].resize(n, m);

      for (j=0; j<n; j++)
	for (k=0; k<m; k++)
	  m_weight[i](j,k) = dis(gen);

      for (j=0; j<n; j++) {
	for (k=0; k<m; k++)
	  std::cerr << "\t" << m_weight[i](j,k);
	std::cerr << "\n";
      }
  }

}

int reseau::forward(const vector & image)
{
  size_t i,j,k,imax,kmax;
  double s;

  for (i=0; i<image.size(); i++) {
    m_neuron_layer[0][i] = image[i]/255.0;
//    std::cerr << "\t\tm_neuron_layer[0][" << i << " ] " << m_neuron_layer[0][i] << std::endl;
  }

  for (j=0; j<m_nlayers-1; j++) {
      imax = m_neuron_layer[j].size();
      kmax = m_neuron_layer[j+1].size();
      for (k=0; k<kmax; k++) {
	  s = m_bias[j][k];
	  for(i=0; i<imax; i++) {
	    s += m_weight[j](k,i) * m_neuron_layer[j][i];
	  }
	  m_neuron_layer[j+1][k] = sigmoid(s/imax);
	  std::cerr << "\t\tm_neuron_layer[" << j+1<<"][" << k<<"] " << m_neuron_layer[j+1][k] << std::endl;
      }
      std::cerr << std::endl;
  }

  int idx = 0;
  double smax = m_neuron_layer[m_nlayers-1][0];
  for (i=1; i<m_neuron_layer[m_nlayers-1].size(); i++) {
      if (fabs(m_neuron_layer[m_nlayers-1][i]) > fabs(smax)) {
	  idx = i;
	  smax = m_neuron_layer[m_nlayers-1][i];
      }
  }

  std::cerr << smax << std::endl;
  return idx;
}
