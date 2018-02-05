#include "scheme.hxx"
#include "parameters.hxx"

#include <sstream>
#include <iomanip>


void Scheme::initialize(const Parameters *P)
{
  m_P = P;
  m_u.init(P);
  m_v.init(P);
  int i;
  for (i=0; i<3; i++) {
    m_n[i] = m_P->n(i);
    m_dx[i] = m_P->dx(i);
    m_di[i] = (m_n[i] < 2) ? 0 : 1;
  }

  kStep = 1;
  m_t = 0.0;

  m_duv = 0.0;

  double dx2 = m_dx[0]*m_dx[0] + m_dx[1]*m_dx[1] + m_dx[2]*m_dx[2];
  m_dt = 0.5*(dx2 + 1e-12);
  m_lambda = 0.25*m_dt/(dx2 + 1e-12);
}

Scheme::~Scheme()
{
}

double Scheme::present()
{
  return m_t;
}

size_t Scheme::getDomainSize(int dim) const
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

bool Scheme::iteration()
{
  int   di = m_di[0],     dj = m_di[1],     dk = m_di[2];
  int i, imin = 1, imax = m_n[0] - 1;
   int j, jmin = 1, jmax = m_n[1] - 1;
   int k, kmin = 1, kmax = m_n[2] - 1;
   double du, du_max = 0.0;

   char s[20];
   sprintf(s, "m_u_%d", m_P->nthreads());
   std::ofstream fu(s);
   m_v.print(fu);
   
   fu << "u(5,5,5) = " << m_u(5,j,k) << std::endl;
   fu << "u(5+di,5,5) = " << m_u(5+di,5,5) << std::endl;
   fu << "u(5-di,5,5) = " << m_u(5-di,5,5) << std::endl;
   
#pragma omp parallel for default(shared), private(i,j,k,du), reduction(+:du_max) 
    for (i = imin; i < imax; i++)
      for (j = jmin; j < jmax; j++)
        for (k = kmin; k < kmax; k++) {
          if (i == 5 && j == 5 && k == 5) {
             fu << "u(i,j,k) = " << m_u(i,j,k) << std::endl;
             fu << "u(i+di,j,k) = " << m_u(i+di,j,k) << std::endl;
             fu << "u(i-di,j,k) = " << m_u(i-di,j,k) << std::endl;
          }
          du = 6 * m_u(i, j, k)
              - m_u(i + di, j, k) - m_u(i - di, j, k)
              - m_u(i, j + dj, k) - m_u(i, j - dj, k)
              - m_u(i, j, k + dk) - m_u(i, j, k - dk);
          du *= m_lambda;
          if (i == 5 && j == 5 && k == 5)
             fu << "m_lambda = " << m_lambda << " du = " << du << std::endl;
          m_v(i, j, k) = m_u(i, j, k) - du;
          du_max += du > 0 ? du : -du;
        }

    sprintf(s, "m_v_%d", m_P->nthreads());
    std::ofstream fv(s);
    m_v.print(fv);
    
    m_duv = du_max;
    exit(-1);
    return true;
}

bool Scheme::solve(unsigned int nSteps)
{
  m_timers[1].start();

  int iStep;

  for (iStep=0; iStep < nSteps; iStep++) {

    m_timers[1].start();
    
    iteration();
    m_u.swap(m_v);
    m_t += m_dt;

    m_timers[1].stop();
    m_timers[2].start();
    std::cerr << " iteration " << std::setw(4) << kStep
              << " variation " << std::setw(12) << std::setprecision(6) << m_duv;
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

double Scheme::variation()
{
  return m_duv;
}

void Scheme::terminate() {
    std::cerr << "\n\nterminate " << codeName << std::endl;
}

const Values & Scheme::getOutput()
{
  return m_u;
}

void Scheme::setInput(const Values & u)
{
  m_u = u;
  m_v = u;
}

void Scheme::save(const char * /*fName*/)
{
}


