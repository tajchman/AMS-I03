#include "scheme.hxx"
#include "parameters.hxx"
#include "version.hxx"
#include <cmath>
#include <sstream>
#include <iomanip>

#ifdef _OPENMP
	#include <omp.h>
#endif

namespace {
	int GetMPICapacity(const Parameters& p)
	{
		int idim = 0;
		int lmin = p.imax(0) - p.imin(0);
		for (int jdim=1; jdim<3; jdim++) {
			int l = p.imax(jdim) - p.imin(jdim);
			if (l < lmin) {
				idim = jdim;
				lmin = l;
			}
		}
		int jdim = (idim+1)%3;
		int kdim = (jdim+1)%3;
		int rmin = p.imin(jdim), rmax = p.imax(jdim);
		int qmin = p.imin(kdim), qmax = p.imax(kdim);
		return (rmax-rmin+1)*(qmax-qmin+1);
	}
}

Scheme::Scheme(Parameters &P, callback_t f) :
    codeName(version), m_P(P), m_u(P), m_v(P)
{
  m_u.init();
  m_v.init();
  m_f = f;
  m_t = 0.0;

  int i;
  for (i=0; i<3; i++) {
    m_dx[i] = m_P.dx(i);
    m_xmin[i] = m_P.xmin(i);
  }

  m_dt = m_P.dt();
  int capacity = GetMPICapacity(m_P);
	m_bufferIn.resize(capacity);
	m_bufferOut.resize(capacity);
}

Scheme::~Scheme()
{
}

double Scheme::present()
{
  return m_t;
}

bool Scheme::iteration()
{
	#ifdef _OPENMP
	int iT = omp_get_thread_num();
	#endif

	double m_duv_threadlocal = iteration_domaine(
		#ifdef _OPENMP
			m_P.imin_thread(0, iT), m_P.imax_thread(0, iT),
			m_P.imin_thread(1, iT), m_P.imax_thread(1, iT),
			m_P.imin_thread(2, iT), m_P.imax_thread(2, iT)
		#else
      m_P.imin(0), m_P.imax(0),
      m_P.imin(1), m_P.imax(1),
      m_P.imin(2), m_P.imax(2)
		#endif
	);
 
	#pragma omp atomic
  m_duv_proclocal += m_duv_threadlocal;

	#pragma omp barrier
	#pragma omp single
	{
		MPI_Allreduce(&m_duv_proclocal, &m_duv, 1, MPI_DOUBLE, MPI_SUM, m_P.comm());
		m_t += m_dt;
    m_u.swap(m_v);
		m_duv_proclocal = 0.0;
	}

  return true;
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

  for (i = imin; i <= imax; i++)
    for (j = jmin; j <= jmax; j++)
      for (k = kmin; k <= kmax; k++) {

        du1 = (-2*m_u(i,j,k) + m_u(i+1,j,k) + m_u(i-1,j,k))*lam_x
            + (-2*m_u(i,j,k) + m_u(i,j+1,k) + m_u(i,j-1,k))*lam_y
            + (-2*m_u(i,j,k) + m_u(i,j,k+1) + m_u(i,j,k-1))*lam_z;

        double x = xmin + i*m_dx[0];
        double y = ymin + j*m_dx[1];
        double z = zmin + k*m_dx[2];
        du2 = m_f({x,y,z});

        du = m_dt * (du1 + du2);
        m_v(i, j, k) = m_u(i, j, k) + du;
        du_sum += du > 0 ? du : -du;
      }

	return du_sum;
}

void Scheme::synchronize()
{	
  for (int idim=0; idim<3; idim++) {

    int jdim = (idim+1)%3;
    int kdim = (jdim+1)%3;

    int omin = m_P.imin(idim), omax = m_P.imax(idim);
    int pmin = m_P.imin(jdim), pmax = m_P.imax(jdim);
    int qmin = m_P.imin(kdim), qmax = m_P.imax(kdim);
    int p, q, m = (pmax-pmin+1)*(qmax-qmin+1);

    std::array<int, 3> i;
    MPI_Status status;

    int voisin = m_P.neighbour(2*idim);
    if (voisin >=0) {
      i[idim] = omin;
			#pragma omp for
      for (p = pmin; p <= pmax; p++) {
				int k = (p - pmin) * (qmax - qmin + 1);
        for (q = qmin; q <= qmax; q++) {
          i[jdim] = p;
					i[kdim] = q;
          m_bufferOut[k++] = m_u(i);
        }
			}

			#pragma omp barrier
			#pragma omp single
			{
				MPI_Sendrecv(m_bufferOut.data(), m, MPI_DOUBLE, voisin, 0,
                   	 m_bufferIn.data(),  m, MPI_DOUBLE, voisin, 0,
                  	 m_P.comm(), &status);
			}

      i[idim] = omin - 1;
      #pragma omp for
      for (p = pmin; p <= pmax; p++) {
				int k = (p - pmin) * (qmax - qmin + 1);
        for (q = qmin; q <= qmax; q++) {
          i[jdim] = p;
					i[kdim] = q;
          m_u(i) = m_bufferIn[k++];
        }
			}
    }

    voisin = m_P.neighbour(2*idim+1);
    if (voisin >=0) {
      i[idim] = omax;
			#pragma omp for
      for (p = pmin; p <= pmax; p++) {
				int k = (p - pmin) * (qmax - qmin + 1);
        for (q=qmin; q<=qmax; q++) {
          i[jdim] = p;
					i[kdim] = q;
          m_bufferOut[k++] = m_u(i);
        }
			}

			#pragma omp barrier
			#pragma omp single
			{
      	MPI_Sendrecv(m_bufferOut.data(), m, MPI_DOUBLE, voisin, 0,
                   	 m_bufferIn.data(),  m, MPI_DOUBLE, voisin, 0,
                  	 m_P.comm(), &status);
			}

      i[idim] = omax + 1;
			#pragma omp for
      for (p = pmin; p <= pmax; p++) {
				int k = (p - pmin) * (qmax - qmin + 1);
        for (q = qmin; q <= qmax; q++) {
          i[jdim] = p;
					i[kdim] = q;
          m_u(i) = m_bufferIn[k++];
        }
			}
    }
  }
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

