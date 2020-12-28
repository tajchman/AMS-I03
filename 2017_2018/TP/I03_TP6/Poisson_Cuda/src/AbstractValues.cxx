/*
 * AbstractValues.cxx
 *
 *  Created on: 12 févr. 2018
 *      Author: marc
 */

#include "AbstractValues.hxx"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>

AbstractValues::AbstractValues(const AbstractParameters * prm)
{
  m_p = prm;
  int i;
  for (i=0; i<3; i++)
    nn *= (m_n[i] = m_p->n(i));

  n1 = m_n[2];      // nombre de points dans la premiere direction
  n2 = m_n[1] * n1; // nombre de points dans le plan des 2 premieres directions
  nn = m_n[0] * n2;
  m_u = NULL;
}

void AbstractValues::swap(AbstractValues & other)
{
  std::swap(m_u, other.m_u);

  int i;
  for (i=0; i<3; i++)
	  std::swap(m_n[i], other.m_n[i]);

  std::swap(n1, other.n1);
  std::swap(n2, other.n2);
}


