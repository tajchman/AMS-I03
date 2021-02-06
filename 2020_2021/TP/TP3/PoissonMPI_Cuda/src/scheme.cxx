#include <cmath>

#include <sstream>
#include <iomanip>

#include "scheme.hxx"
#include "parameters.hxx"
#include "version.hxx"
#include "iteration.hxx"
#include "variation.hxx"
#include "timer_id.hxx"
#include "dim.hxx"


Scheme::Scheme(Parameters &P) :
    codeName(version), m_P(P), m_u(P), m_v(P)  {

  m_t = 0.0;
  m_duv = 0.0;

  double lx[3];
  int i;
  for (i=0; i<3; i++) {
    m_n[i] = m_P.n(i);
    m_dx[i] = m_P.dx(i);
    m_xmin[i] = m_P.xmin(i);
    lx[i] = 1.0/(m_dx[i]*m_dx[i]);
  }

  m_dt = m_P.dt();

  setDims(m_n, m_xmin, m_dx, lx);

  diff = NULL;
  partialDiff = NULL;
}

Scheme::~Scheme()
{
  freeVariationData(diff, partialDiff);
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
  m_u.synchronized(false);

  T.stop();
}

void Scheme::synchronize()
{

  for (int idim=0; idim<3; idim++) {

    int jdim = (idim+1)%3;
    int kdim = (jdim+1)%3;

    int omin = m_P.imin(idim), omax = m_P.imax(idim);
    int pmin = m_P.imin(jdim), pmax = m_P.imax(jdim);
    int qmin = m_P.imin(kdim), qmax = m_P.imax(kdim);
    int k, p, q, m = (pmax-pmin+1)*(qmax-qmin+1);

    int i[3];
    std::vector<double> bufferIn(m), bufferOut(m);
    MPI_Status status;

    int voisin = m_P.neighbour(2*idim);
    if (voisin >=0) {
      i[idim] = omin;
      for (k=0, p=pmin; p<=pmax; p++)
        for (q=qmin; q<=qmax; q++, k++) {
          i[jdim] = p; i[kdim] = q;
          bufferOut[k] = m_u(i[0], i[1], i[2]);
        }

      MPI_Sendrecv(bufferOut.data(), m, MPI_DOUBLE, voisin, 0,
                   bufferIn.data(),  m, MPI_DOUBLE, voisin, 0,
                   m_P.comm(), &status);

      i[idim] = omin - 1;
      for (k=0, p=pmin; p<=pmax; p++)
        for (q=qmin; q<=qmax; q++, k++) {
          i[jdim] = p; i[kdim] = q;
          m_u(i[0], i[1], i[2]) = bufferIn[k];
        }
    }

    voisin = m_P.neighbour(2*idim+1);
    if (voisin >=0) {
      i[idim] = omax;
      for (k=0, p=pmin; p<=pmax; p++)
        for (q=qmin; q<=qmax; q++, k++) {
          i[jdim] = p; i[kdim] = q;
          bufferOut[k] = m_u(i[0], i[1], i[2]);
        }

      MPI_Sendrecv(bufferOut.data(), m, MPI_DOUBLE, voisin, 0,
                   bufferIn.data(),  m, MPI_DOUBLE, voisin, 0,
                   m_P.comm(), &status);

      i[idim] = omax + 1;
      for (k=0, p=pmin; p<=pmax; p++)
        for (q=qmin; q<=qmax; q++, k++) {
          i[jdim] = p; i[kdim] = q;
          m_u(i[0], i[1], i[2]) = bufferIn[k];
        }
    }
  }
}

double Scheme::iteration_domaine(int imin, int imax,
                                 int jmin, int jmax,
                                 int kmin, int kmax)
{
  iterationWrapper(m_v, m_u, m_dt, m_n, 
                   imin, imax, jmin, jmax, kmin, kmax);

  m_v.synchronized(false);
  return variationWrapper(m_u, m_v, 
                          diff, partialDiff, 
                          m_n[0]*m_n[1]*m_n[2]);
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

