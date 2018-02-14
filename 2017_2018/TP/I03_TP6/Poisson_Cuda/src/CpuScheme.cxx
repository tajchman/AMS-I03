#include "CpuScheme.hxx"
#include "CpuValues.hxx"
#include "CpuParameters.hxx"

#include <sstream>
#include <iomanip>

CpuScheme::CpuScheme(const CpuParameters *P) : AbstractScheme(P) {

  m_u = new CpuValues(P);
  m_v = new CpuValues(P);

  codeName = "Poisson_CPU";
  deviceName = "CPU";
}

CpuScheme::~CpuScheme()
{
  delete m_u;
  delete m_v;
}


bool CpuScheme::iteration()
{
  int   di = m_di[0],     dj = m_di[1],     dk = m_di[2];
  int i, j, k;
  double du, du_max;

  int imin = m_P->imin(0) ;
  int jmin = m_P->imin(1) ;
  int kmin = m_P->imin(2) ;

  int imax = m_P->imax(0) ;
  int jmax = m_P->imax(1) ;
  int kmax = m_P->imax(2) ;

  CpuValues * pu = dynamic_cast<CpuValues *>(m_u);
  CpuValues & u = *pu;

  CpuValues * pv = dynamic_cast<CpuValues *>(m_v);
  CpuValues & v = *pv;

  du_max = 0.0;
    
  for (i = imin; i < imax; i++)
    for (j = jmin; j < jmax; j++)
      for (k = kmin; k < kmax; k++) { 
	du = 6 * u(i, j, k)
	  - u(i + di, j, k) - u(i - di, j, k)
	  - u(i, j + dj, k) - u(i, j - dj, k)
	  - u(i, j, k + dk) - u(i, j, k - dk);
        du *= m_lambda;
        v(i, j, k) = u(i, j, k) - du;
        du_max += du > 0 ? du : -du;
      }

  m_duv_max = du_max;
  return true;
}


