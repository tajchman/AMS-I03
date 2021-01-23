#include "scheme.hxx"
#include "parameters.hxx"
#include "version.hxx"
#include <cmath>
#include <sstream>
#include <iomanip>

#include "dim.cuh"
#include "variation.cuh"

Scheme::Scheme(Parameters &P) :
    codeName(version), m_P(P), m_u(P), m_v(P)  {

  m_u.init();
  m_v.init();
  m_t = 0.0;
  m_duv = 0.0;

  double lx[3];
  int i;
  for (i=0; i<3; i++) {
    m_n[i] = m_P.imax(i) - m_P.imin(i) + 3;
    m_dx[i] = m_P.dx(i);
    m_xmin[i] = m_P.xmin(i);
    lx[i] = 1.0/(m_dx[i]*m_dx[i]);
  }

  m_dt = m_P.dt();

  cudaMemcpyToSymbol(n, &m_n, 3 * sizeof(int));
  cudaMemcpyToSymbol(xmin, &m_xmin, 3 * sizeof(double));
  cudaMemcpyToSymbol(dx, &m_dx, 3 * sizeof(double));
  cudaMemcpyToSymbol(lambda, &lx, 3 * sizeof(double));

  diff = NULL;
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

  m_duv = iteration_domaine(
      m_P.imin(0), m_P.imax(0),
      m_P.imin(1), m_P.imax(1),
      m_P.imin(2), m_P.imax(1));

  m_t += m_dt;
  m_u.swap(m_v);
}

__device__
double f(double x, double y, double z)
{
  if (x < 0.3)
    return 0.0;
  else
    return sin(x-0.5) * cos(y-0.5) * exp(- z*z);
}

__global__
void iterCuda(double *u, double *v, double dt)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  int p;
  double du, du1, du2, x, y, z;

  if (i<n[0] && j<n[1] && k<n[2]) {
    p = i + n[0] * (j + k*n[1]);
    du1 = (- 2*u[p] + u[p + 1]         + u[p - 1])*lambda[0]
        + (- 2*u[p] + u[p + n[0]]      + u[p - n[0]])*lambda[1]
        + (- 2*u[p] + u[p + n[0]*n[1]] + u[p - n[0]*n[1]])*lambda[2];
        
    x = xmin[0] + i*dx[0];
    y = xmin[1] + j*dx[1];
    z = xmin[2] + k*dx[2];

    du2 = f(x,y,z);
        
    du = dt * (du1 + du2);
    v[p] = u[p] + du;
  }
}

double Scheme::iteration_domaine(int imin, int imax, 
                                 int jmin, int jmax,
                                 int kmin, int kmax)
{
  dim3 dimBlock(8,8,8);

  dim3 dimGrid(ceil(m_n[0]/double(dimBlock.x)), 
               ceil(m_n[1]/double(dimBlock.y)),
               ceil(m_n[2]/double(dimBlock.z)));

  iterCuda<<<dimGrid, dimBlock>>>(m_u.dataGPU(), m_v.dataGPU(), m_dt);

  return variationCuda(m_u.dataGPU(), m_v.dataGPU(), 
                       diff, m_n[0]*m_n[1]*m_n[2]);
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


