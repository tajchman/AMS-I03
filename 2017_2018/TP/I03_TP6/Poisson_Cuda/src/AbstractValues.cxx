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
}

void AbstractValues::operator=(AbstractValues const& other)
{
   allocate(nn);
 
}

void AbstractValues::swap(AbstractValues & other)
{
  double * dtemp = m_u;
  m_u = other.m_u;
  other.m_u = dtemp;

  int i, temp;
  for (i=0; i<3; i++) {
    temp = m_n[i];
    m_n[i] = other.m_n[i];
    other.m_n[i] = temp;
  }
}


