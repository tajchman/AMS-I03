/*
 * AbstractScheme.cxx
 *
 *  Created on: 12 févr. 2018
 *      Author: marc
 */

#include "AbstractScheme.hxx"
#include <iomanip>
#include <iostream>

AbstractScheme::AbstractScheme(const Parameters *P) : m_timers(3)  {
   m_timers[0].name("init");
   m_timers[1].name("solve");
   m_timers[2].name("other");
   m_duv_max = 0.0;
   m_P = P;
   m_t = 0.0;
   kStep = 0;
   m_dt = 0.0;
   m_lambda = 0.0;

   int i;
   for (i=0; i<3; i++) {
     m_n[i] = m_P->n(i);
     m_dx[i] = m_P->dx(i);
     m_di[i] = (m_n[i] < 2) ? 0 : 1;
   }

   double dx2 = m_dx[0]*m_dx[0] + m_dx[1]*m_dx[1] + m_dx[2]*m_dx[2];
   m_dt = 0.5*(dx2 + 1e-12);
   m_lambda = 0.25*m_dt/(dx2 + 1e-12);

   m_u = NULL;
   m_v = NULL;
   codeName = "";
   deviceName = "";
}

void AbstractScheme::initialize()
{
  m_u->init();
  m_v->init();

  kStep = 1;
  m_t = 0.0;

  m_duv_max = 0.0;

  double dx2 = m_dx[0]*m_dx[0] + m_dx[1]*m_dx[1] + m_dx[2]*m_dx[2];
  m_dt = 0.5*(dx2 + 1e-12);
  m_lambda = 0.25*m_dt/(dx2 + 1e-12);
}

AbstractScheme::~AbstractScheme()
{
}

double AbstractScheme::present()
{
  return m_t;
}

size_t AbstractScheme::getDomainSize(int dim) const
{
  size_t d;
  switch (dim) {
    case 0:
      d = m_n[0];
      break;
    case 1:
      d = m_n[1];
      break;
    case 2:
      d = m_n[2];
      break;
    default:
      d = 1;
  }
  return d;
}


bool AbstractScheme::solve(unsigned int nSteps)
{

  int iStep;

  for (iStep=0; iStep < nSteps; iStep++) {

    m_timers[1].start();

    iteration();

    m_t += m_dt;

    m_u->swap(*m_v);

    m_timers[1].stop();
    m_timers[2].start();
    std::cerr << " iteration " << std::setw(4) << kStep
              << " variation " << std::setw(12) << std::setprecision(6)
	      << m_duv_max;
    size_t i, n = m_timers.size();
    std::cerr << " (times :";
    for (i=0; i<n; i++)
      std::cerr << " " << std::setw(5) << m_timers[i].name()
	        << " " << std::setw(9) << std::fixed << m_timers[i].elapsed();
    std::cerr	  << ")   \n";
    m_timers[2].stop();

    kStep++;
  }

  return true;
}

double AbstractScheme::variation()
{
  return m_duv_max;
}

void AbstractScheme::terminate() {
    std::cerr << "\n\nterminate " << codeName << std::endl;
}

const AbstractValues & AbstractScheme::getOutput()
{
  return *m_u;
}

void AbstractScheme::setInput(const AbstractValues & u)
{
  *m_u = u;
  *m_v = u;
}

void AbstractScheme::save(const char * /*fName*/)
{
}
