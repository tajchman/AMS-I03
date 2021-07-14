#include "scheme.hxx"
#include "parameters.hxx"
#include "version.hxx"
#include "user.hxx"
#include "timer_id.hxx"
#include <cmath>

#include <sstream>
#include <iomanip>


Scheme::Scheme(Parameters &P) :
    codeName(version), m_P(P), m_u(P), m_v(P)  {

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

Scheme::~Scheme()
{
}

double Scheme::present()
{
  return m_t;
}

void Scheme::iteration()
{

  Timer & T = GetTimer(T_IterationId); T.start();

  m_duv = iteration_domaine(
      m_P.imin(0), m_P.imax(0),
      m_P.imin(1), m_P.imax(1),
      m_P.imin(2), m_P.imax(2));

  m_t += m_dt;
  m_u.swap(m_v);

  T.stop();
}

double Scheme::iteration_domaine(int imin, int imax,
                                 int jmin, int jmax,
                                 int kmin, int kmax)
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

  #pragma omp parallel for reduction(+:du_sum) firstprivate(lam_x, lam_y, lam_z, xmin, ymin, zmin, imin, jmin, kmin, imax, jmax, kmax) private(x,y,z,i,j,k,du,du1,du2) default(none)
  for (k = kmin; k <= kmax; k++) 
    for (j = jmin; j <= jmax; j++) {
      double * p00 = m_u.line(j,k);
      double * q00 = m_v.line(j,k);
      double * pm0 = m_u.line(j-1,k);
      double * pp0 = m_u.line(j+1,k);
      double * p0m = m_u.line(j,k-1);
      double * p0p = m_u.line(j,k+1);
      
      for (i = imin; i <= imax; i++, p00++, q00++, pm0++, pp0++, p0m++, p0p++)
      {
        double u00 = p00[0];
        du1 = (-2*u00 + p00[1] + p00[-1])*lam_x
            + (-2*u00 + pp0[0] + pm0[0])*lam_y
            + (-2*u00 + p0p[0] + p0m[0])*lam_z;

        x = xmin + i*m_dx[0];
        y = ymin + j*m_dx[1];
        z = zmin + k*m_dx[2];
        du2 = force(x,y,z);

        du = m_dt * (du1 + du2);
        q00[0] = u00 + du;
        du_sum += du > 0 ? du : -du;
      }
    }
    return 0.0;
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


