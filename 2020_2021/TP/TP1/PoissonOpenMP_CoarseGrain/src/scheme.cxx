#include "scheme.hxx"
#include "parameters.hxx"
#include "version.hxx"

#include <sstream>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

Scheme::Scheme(Parameters &P, callback_t f) :
    codeName(version), m_P(P), m_u(P), m_v(P)  {

  m_u.init();
  m_v.init();
  m_f = f;
  m_t = 0.0;
  m_duv = 0.0;

  int i;
  for (i=0; i<3; i++) {
    m_n[i] = m_P.n(i);
    m_dx[i] = m_P.dx(i);
    m_di[i] = (m_n[i] < 2) ? 0 : 1;
  }

  m_dt = m_P.dt();
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
 #pragma omp single
 m_duv = 0.0;

#ifdef _OPENMP
  int iThread = omp_get_thread_num();
#else
  int iThread = 0;
#endif

  double du_sum = iteration_domaine(
    m_P.imin_local(0, iThread), m_P.imax_local(0, iThread), 
    m_P.imin_local(1, iThread), m_P.imax_local(1, iThread), 
    m_P.imin_local(2, iThread), m_P.imax_local(2, iThread));
  
#ifdef DEBUG
  #pragma omp critical
  {
   std::cerr << "du_sum : " << du_sum << std::endl;
  }
#endif


  #pragma omp atomic
  m_duv += du_sum;

  #pragma omp single
  {
  m_t += m_dt;
  m_u.swap(m_v);
  }

  return true;
}

double Scheme::iteration_domaine(int imin, int imax, 
                                 int jmin, int jmax,
                                 int kmin, int kmax)
{
#ifdef DEBUG
  #pragma omp critical
  {
#ifdef _OPENMP
    int iThread = omp_get_thread_num();
#else
    int iThread = 0;
#endif
    std::cerr << "iteration Thread " << iThread 
              << "   [" << imin << "," << imax <<")"
              << " x [" << jmin << "," << jmax <<")"
              << " x [" << kmin << "," << kmax <<")" 
              << std::endl;
  }
#endif

  double lam_x = 1/(m_dx[0]*m_dx[0]);
  double lam_y = 1/(m_dx[1]*m_dx[1]);
  double lam_z = 1/(m_dx[2]*m_dx[2]);
  double xmin =  m_P.xmin(0);
  double ymin =  m_P.xmin(1);
  double zmin =  m_P.xmin(2);

  int i,j,k;
  int   di = m_di[0],     dj = m_di[1],     dk = m_di[2];
  double du, du1, du2, du_sum = 0.0;
  
  for (i = imin; i < imax; i++)
    for (j = jmin; j < jmax; j++)
      for (k = kmin; k < kmax; k++) {
        du1 = (-2*m_u(i,j,k) + m_u(i+di,j,k) + m_u(i-di,j,k))*lam_x
           + (-2*m_u(i,j,k) + m_u(i,j+dj,k) + m_u(i,j-dj,k))*lam_y
           + (-2*m_u(i,j,k) + m_u(i,j,k+dk) + m_u(i,j,k-dk))*lam_z;

        double x = xmin + i*m_dx[0];
        double y = xmin + j*m_dx[1];
        double z = xmin + k*m_dx[2];
        du2 = m_f(x,y,z);

        du = m_dt * (du1 + du2);
        m_v(i, j, k) = m_u(i, j, k) + du;
        du_sum += du > 0 ? du : -du;
      }

    return du_sum;
}

void Scheme::initialize() {
  std::cerr << "\nintialize " << codeName << std::endl;
}

void Scheme::terminate() {
  std::cerr << "\nterminate " << codeName << std::endl;
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


