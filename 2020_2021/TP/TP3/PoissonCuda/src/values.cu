#include "values.hxx"
#include "os.hxx"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include "cuda_check.cuh"
#include "dim.cuh"

Values::Values(Parameters & prm) : m_p(prm)
{
  int i;

  for (i=0; i<3; i++) {
    m_imin[i] = m_p.imin(i);
    m_imax[i] = m_p.imax(i);
    m_n_local[i] = m_imax[i] - m_imin[i] + 3;
    m_dx[i] = m_p.dx(i);
    m_xmin[i] = m_p.xmin(i);
    m_xmax[i] = m_p.xmax(i);
    nn *= m_n_local[i];
  }

  n1 = m_n_local[0];      // nombre de points dans la premiere direction
  n2 = m_n_local[1] * n1; // nombre de points dans le plan des 2 premieres directions
  nn = n2 * m_n_local[2];
  std::cerr << "nn = " << nn << std::endl;
  CUDA_CHECK_OP(cudaMalloc(&m_u, nn*sizeof(double)));
  h_u = new double[nn];
  h_synchronized = false;

  zero();
}

Values::~Values()
{
  delete [] h_u;
  cudaFree(m_u);
}

__global__
void zeroValue(int n, double *u)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i<n) {
    u[i] = 0.0;
  }
}

void Values::zero()
{
  int dimBlock = 256;
  int dimGrid = (nn + dimBlock - 1)/dimBlock;

  std::cerr << "dimGrid " << dimGrid << std::endl;
  zeroValue<<<dimGrid, dimBlock>>>(nn, m_u);
  CUDA_CHECK_KERNEL();
  h_synchronized = false;
}


__device__
double cond_ini(double x, double y, double z)
{
  return floor(((x-0.5)*(x-0.5) 
              + (y-0.5)*(y-0.5)
              + (z-0.5)*(z-0.5))/0.1);
}

__device__
double cond_lim(double x, double y, double z)
{
  return 1.0;
}

__global__
void initValue(double *u)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  int p;

  if (i<n[0] && j<n[1] && k<n[2]) {
    p = i + j*n[0] + k*n[0]*n[1];
    u[p] = cond_ini(xmin[0] + i*dx[0],
                    xmin[1] + j*dx[1], 
                    xmin[2] + k*dx[2]);
  }
}

void Values::init()
{
  dim3 dimBlock(8,8,8);
  dim3 dimGrid(ceil(m_n_local[0]/double(dimBlock.x)),
               ceil(m_n_local[1]/double(dimBlock.y)),
               ceil(m_n_local[2]/double(dimBlock.z)));

  initValue<<<dimGrid, dimBlock>>>(m_u);
  CUDA_CHECK_KERNEL();
}

__global__
void boundZValue(int k, double *u)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int p;

  if (i<n[0] && j<n[1]) {
    p = i + j*n[0] + k*n[0]*n[1];
    u[p] = cond_lim(xmin[0] + i*dx[0], 
                    xmin[1] + j*dx[1], 
                    xmin[2] + k*dx[2]);
  }
}

__global__
void boundYValue(int j, double *u)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int k = threadIdx.y + blockIdx.y * blockDim.y;
  int p;

  if (i<n[0] && k<n[2]) {
    p = i + j*n[0] + k*n[0]*n[1];
    u[p] = cond_lim(xmin[0] + i*dx[0], 
                    xmin[1] + j*dx[1], 
                    xmin[2] + k*dx[2]);
  }
}

__global__
void boundXValue(int i, double *u)
{
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int k = threadIdx.y + blockIdx.y * blockDim.y;
  int p;

  if (j<n[1] && k<n[2]) {
    p = i + j*n[0] + k*n[0]*n[1];
    u[p] = cond_lim(xmin[0] + i*dx[0], 
                    xmin[1] + j*dx[1], 
                    xmin[2] + k*dx[2]);
  }
}

void Values::boundaries()
{
  dim3 dimBlock(16,16);

  dim3 dimGrid2(ceil(m_n_local[0]/double(dimBlock.x)), ceil(m_n_local[1]/double(dimBlock.y)));
  boundZValue<<<dimGrid2, dimBlock>>>(m_imin[2], m_u);
  boundZValue<<<dimGrid2, dimBlock>>>(m_imax[2], m_u);

  dim3 dimGrid1(ceil(m_n_local[0]/double(dimBlock.x)), ceil(m_n_local[2]/double(dimBlock.y)));
  boundYValue<<<dimGrid1, dimBlock>>>(m_imin[1], m_u);
  boundYValue<<<dimGrid1, dimBlock>>>(m_imax[1], m_u);

  dim3 dimGrid0(ceil(m_n_local[1]/double(dimBlock.x)), ceil(m_n_local[2]/double(dimBlock.y)));
  boundZValue<<<dimGrid0, dimBlock>>>(m_imin[0], m_u);
  boundZValue<<<dimGrid0, dimBlock>>>(m_imax[0], m_u);
  CUDA_CHECK_KERNEL();

  h_synchronized = false;
}


std::ostream & operator<< (std::ostream & f, const Values & v)
{
  v.print(f);
  return f;
}

void Values::print(std::ostream & f) const
{
    int i, j, k;
    
    if (!h_synchronized) {
      cudaMemcpy(h_u, m_u, nn, cudaMemcpyDeviceToHost);
      h_synchronized = true;
    }

    for (i=m_imin[0]; i<=m_imax[0]; i++) {
      for (j=m_imin[1]; j<=m_imax[1]; j++) {
        for (k=m_imin[2]; k<=m_imax[2]; k++) {
          f << " " << operator()(i,j,k);
        }
        f << std::endl;
      }
        f << std::endl;
    }
}

template<typename T>
void swap(T & a, T & b)
{
  T t = a;
  a = b;
  b = t;
}

void Values::swap(Values & other)
{
  ::swap(m_u, other.m_u);
  int i;
  for (i=0; i<3; i++) {
    ::swap(m_imin[i], other.m_imin[i]);
    ::swap(m_imax[i], other.m_imax[i]);
    ::swap(m_n_local[i], other.m_n_local[i]);
    ::swap(m_dx[i], other.m_dx[i]);
    ::swap(m_xmin[i], other.m_xmin[i]);
    ::swap(m_xmax[i], other.m_xmax[i]);
  }
}

void Values::plot(int order) const {

  std::ostringstream s;
  int i, j, k;
  int imin = m_p.imin(0);
  int jmin = m_p.imin(1);
  int kmin = m_p.imin(2);

  int imax = m_p.imax(0);
  int jmax = m_p.imax(1);
  int kmax = m_p.imax(2);

  s << m_p.resultPath();
  s << "/0";
  mkdir_p(s.str().c_str());
  
  s << "/plot_" << order << ".vtr";
  std::ofstream f(s.str().c_str());

  f << "<?xml version=\"1.0\"?>\n";
  f << "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
    << "<RectilinearGrid WholeExtent=\""
    << imin << " " << imax-1  << " " 
    << jmin << " " << jmax-1  << " " 
    << kmin << " " << kmax-1 
    << "\">\n"
    << "<Piece Extent=\""
    << imin << " " << imax-1  << " " 
    << jmin << " " << jmax-1  << " " 
    << kmin << " " << kmax-1 
    << "\">\n";

  f << "<PointData Scalars=\"values\">\n";
  f << "  <DataArray type=\"Float64\" Name=\"values\" format=\"ascii\">\n";
  
  for (k=kmin; k<kmax; k++)
    for (j=jmin; j<jmax; j++) {
      for (i=imin; i<imax; i++)
//        f << " " << operator()(i,j,k);
      f << "\n";
    }
  f << " </DataArray>\n";
   
  f << "</PointData>\n";

  f << " <Coordinates>\n";
  
  for (k=0; k<3; k++) {
    f << "   <DataArray type=\"Float64\" Name=\"" << char('X' + k) << "\"" 
      << " format=\"ascii\">";
    
    int imin = m_p.imin(k);
    int imax = m_p.imax(k);
    for (i=imin; i<imax; i++)
      f << " " << i * m_p.dx(k);
    f << "   </DataArray>\n";
  }
  f << " </Coordinates>\n";
  
  f << "</Piece>\n"
    << "</RectilinearGrid>\n"
    << "</VTKFile>\n" <<std::endl;
}

void Values::operator= (const Values &other)
{
  int i;

  for (i=0; i<3; i++) {
    m_imin[i] = other.m_imin[i];
    m_imax[i] = other.m_imax[i];
    m_n_local[i] = other.m_n_local[i];
    m_xmin[i] = other.m_xmin[i];
    m_xmax[i] = other.m_xmax[i];
    m_dx[i] = other.m_dx[i];
  }
  m_u = other.m_u;
}