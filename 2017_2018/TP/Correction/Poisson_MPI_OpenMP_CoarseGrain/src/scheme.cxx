#include "scheme.hxx"
#include "parameters.hxx"

#include <mpi.h>
#include <sstream>
#include <iomanip>
#if defined(_OPENMP)
#include <omp.h>
#endif

Scheme::Scheme(const Parameters *P) :
  codeName("Poisson_MPI_OpenMP_CoarseGrain"), m_u(P), m_v(P), m_timers(4)  {
  m_timers[0].name("init");
  m_timers[1].name("solve");
  m_timers[2].name("comm");
  m_timers[3].name("other");
  m_duv = 0.0;
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
}

void Scheme::initialize()
{
  m_u.init();
  m_v.init();

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

double Scheme::iteration()
{
  int   di = m_di[0],     dj = m_di[1],     dk = m_di[2];
  int i, j, k;
  double du, du_max;

  int ith = omp_get_thread_num();

  int imin = m_P->thread_imin(0, ith) ;
  int jmin = m_P->thread_imin(1, ith) ;
  int kmin = m_P->thread_imin(2, ith) ;

  int imax = m_P->thread_imax(0, ith) ;
  int jmax = m_P->thread_imax(1, ith) ;
  int kmax = m_P->thread_imax(2, ith) ;

  du_max = 0.0;
    
  for (i = imin; i < imax; i++)
    for (j = jmin; j < jmax; j++)
      for (k = kmin; k < kmax; k++) {
        du = 6 * m_u(i, j, k)
          - m_u(i + di, j, k) - m_u(i - di, j, k)
          - m_u(i, j + dj, k) - m_u(i, j - dj, k)
          - m_u(i, j, k + dk) - m_u(i, j, k - dk);
        du *= m_lambda;
        m_v(i, j, k) = m_u(i, j, k) - du;
        du_max += du > 0 ? du : -du;
      }

  return du_max;
}

bool Scheme::solve(unsigned int nSteps)
{

  int iStep;

  for (iStep=0; iStep < nSteps; iStep++) {

#pragma omp master
    {
      m_timers[2].start();
      m_u.synchronize();
      m_timers[2].stop();
    
      m_timers[1].start();
      m_duv = 0.0;
    }
#pragma omp barrier

    double du_partiel = iteration();

#pragma omp atomic
    m_duv += du_partiel;
    
#pragma omp barrier

#pragma omp master
    {
      double du_max_global;
      MPI_Allreduce(&m_duv, &du_max_global, 1, MPI_DOUBLE, MPI_SUM, m_P->comm());
      m_duv = du_max_global;
      m_t += m_dt;

      m_u.swap(m_v);

      m_timers[1].stop();
      if (m_P->rank() == 0) {
	m_timers[3].start();
	std::cerr << " iteration " << std::setw(4) << kStep
		  << " variation " << std::setw(12) << std::setprecision(6) << m_duv;
	size_t i, n = m_timers.size();
	std::cerr << " (times :";
	for (i=0; i<n; i++)
	  std::cerr << " " << std::setw(5) << m_timers[i].name()
		    << " " << std::setw(9) << std::fixed << m_timers[i].elapsed();
	std::cerr	  << ")   \n";
	m_timers[3].stop();
      }

      kStep++;
    }
  }

  return true;
}

double Scheme::variation()
{
  return m_duv;
}

void Scheme::terminate() {
  if (m_P->rank() == 0)
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


