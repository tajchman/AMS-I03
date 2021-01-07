#include "scheme.hxx"
#include "parameters.hxx"
#include "version.hxx"
#include <cmath>

#include <sstream>
#include <iomanip>

#if defined(_OPENMP)
   #include <omp.h>
#endif

Scheme::Scheme(Parameters &P, callback_t f)
  : codeName(version), m_P(P), m_u(P), m_v(P)
{
  m_f = f;
  m_t = 0.0;
  m_duv = 0.0;

  int i;
  for (i=0; i<3; i++) {
    m_n[i] = m_P.n(i);
    m_dx[i] = m_P.dx(i);
    m_xmin[i] = m_P.xmin(i);
  }

  m_dt = m_P.dt();
}

double Scheme::present() {
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

#ifdef _OPENMP
  int iT = omp_get_thread_num(); 
  int imin_loc = m_P.imin_local(0, iT);
  int imax_loc = m_P.imax_local(0, iT);
  int jmin_loc = m_P.imin_local(1, iT);
  int jmax_loc = m_P.imax_local(1, iT);
  int kmin_loc = m_P.imin_local(2, iT);
  int kmax_loc = m_P.imax_local(2, iT);
#else
  int iT = 0;
  int imin_loc = m_P.imin(0);
  int imax_loc = m_P.imax(0);
  int jmin_loc = m_P.imin(1);
  int jmax_loc = m_P.imax(1);
  int kmin_loc = m_P.imin(2);
  int kmax_loc = m_P.imax(2);
#endif

//  std::ostringstream out;
//  out << "Thread num: " << iT << "\n";
//  out << imin_loc << " " << imax_loc << "\n";
//  out << jmin_loc << " " << jmax_loc << "\n";
//  out << kmin_loc << " " << kmax_loc << "\n";
//  std::cout << out.str();

  double m_duv_loc = iteration_domaine(
    imin_loc, imax_loc,
    jmin_loc, jmax_loc,
    kmin_loc, kmax_loc);
 
  #pragma omp barrier

  #pragma omp single
  {
    m_duv = 0.;
    m_t += m_dt;
    m_u.swap(m_v);
  }
  #pragma omp critical
  {
    m_duv += m_duv_loc;
  }

  #pragma omp barrier 

  return true;
}

double Scheme::iteration_domaine(int imin_, int imax_, 
                                 int jmin_, int jmax_,
                                 int kmin_, int kmax_)
{
  double lam_x = 1/(m_dx[0]*m_dx[0]);
  double lam_y = 1/(m_dx[1]*m_dx[1]);
  double lam_z = 1/(m_dx[2]*m_dx[2]);
  double xmin = m_xmin[0];
  double ymin = m_xmin[1];
  double zmin = m_xmin[2];
  int i,j,k;
  double du, du1, du2, du_sum = 0.0;
  
  double x, y, z;

  double correction = 0.0;

  for (i = imin_; i < imax_; i++)
    for (j = jmin_; j < jmax_; j++)
      for (k = kmin_; k < kmax_; k++) {
           
        du1 = (-2*m_u(i,j,k) + m_u(i+1,j,k) + m_u(i-1,j,k))*lam_x
            + (-2*m_u(i,j,k) + m_u(i,j+1,k) + m_u(i,j-1,k))*lam_y
            + (-2*m_u(i,j,k) + m_u(i,j,k+1) + m_u(i,j,k-1))*lam_z;

        x = xmin + i*m_dx[0];
        y = ymin + j*m_dx[1];
        z = zmin + k*m_dx[2];
        du2 = m_f(x,y,z);

        du = m_dt * (du1 + du2);
        m_v(i, j, k) = m_u(i, j, k) + du;
        du = du > 0 ? du : -du;

        double corrected = du - correction;
        double new_sum = du_sum + corrected;
        correction = (new_sum - du_sum) - corrected;
        du_sum = new_sum;
      }

    return du_sum - correction;
}

const Values& Scheme::getOutput()
{
  return m_u;
}

void Scheme::setInput(Values&& u)
{
  m_u = std::move(u);
  m_v.resize(m_u.size(0), m_u.size(1), m_u.size(2));
}
