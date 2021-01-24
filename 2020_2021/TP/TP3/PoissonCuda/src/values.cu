#include "values.hxx"
#include "os.hxx"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include "cuda_check.cuh"
#include "dim.cuh"
#include "timer_id.hxx"

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

  Timer & T = GetTimer(T_AllocId); T.start();

  CUDA_CHECK_OP(cudaMalloc(&d_u, nn*sizeof(double)));
  h_u = new double[nn];

  T.stop();

  h_synchronized = false;

  Timer & Ti = GetTimer(T_InitId); Ti.start();

  zero();

  Ti.stop();
}

Values::~Values()
{
  Timer & T = GetTimer(T_FreeId); T.start();
  delete [] h_u;
  cudaFree(d_u);
  T.stop();
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

  zeroValue<<<dimGrid, dimBlock>>>(nn, d_u);
  CUDA_CHECK_KERNEL();
  h_synchronized = false;
}


__device__
double cond_ini(double x, double y, double z)
{
  double xc = x - 0.5;
  double yc = y - 0.5;
  double zc = z - 0.5;

  double c = xc*xc+yc*yc+zc*zc;
  return (c < 0.1) ? 1.0 : 0.0;
}

__global__
void initValue(double *u)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  int p;

  if (i>0 && i<n[0]-1 && j>0 && j<n[1]-1 && k>0 && k<n[2]-1)
  {
    p = i + n[0] * (j + k*n[1]);
    u[p] = cond_ini(xmin[0] + i*dx[0],
                    xmin[1] + j*dx[1], 
                    xmin[2] + k*dx[2]);
  }
}

void Values::init()
{
  Timer & T = GetTimer(T_InitId); T.start();

  dim3 dimBlock(8,8,8);
  dim3 dimGrid(ceil(m_n_local[0]/double(dimBlock.x)),
               ceil(m_n_local[1]/double(dimBlock.y)),
               ceil(m_n_local[2]/double(dimBlock.z)));

  initValue<<<dimGrid, dimBlock>>>(d_u);
  cudaDeviceSynchronize();
  CUDA_CHECK_KERNEL();
  
  h_synchronized = false;
  T.stop();
}

__device__
double cond_lim(double x, double y, double z)
{
  return 0.0;
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
  int k = threadIdx.z + blockIdx.z * blockDim.z;
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
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
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
  Timer & T = GetTimer(T_InitId); T.start();

  dim3 dimBlock2(16,16,1);
  dim3 dimGrid2(ceil(m_n_local[0]/double(dimBlock2.x)), ceil(m_n_local[1]/double(dimBlock2.y)), 1);
  boundZValue<<<dimGrid2, dimBlock2>>>(m_imin[2]-1, d_u);
  boundZValue<<<dimGrid2, dimBlock2>>>(m_imax[2]+1, d_u);

  dim3 dimBlock1(16,1,16);
  dim3 dimGrid1(ceil(m_n_local[0]/double(dimBlock1.x)), 1, ceil(m_n_local[2]/double(dimBlock1.z)));
  boundYValue<<<dimGrid1, dimBlock1>>>(m_imin[1]-1, d_u);
  boundYValue<<<dimGrid1, dimBlock1>>>(m_imax[1]+1, d_u);

  dim3 dimBlock0(1,16,16);
  dim3 dimGrid0(1, ceil(m_n_local[1]/double(dimBlock0.y)), ceil(m_n_local[2]/double(dimBlock0.z)));
  boundXValue<<<dimGrid0, dimBlock0>>>(m_imin[0]-1, d_u);
  boundXValue<<<dimGrid0, dimBlock0>>>(m_imax[0]+1, d_u);

  cudaDeviceSynchronize();
  CUDA_CHECK_KERNEL();

  T.stop();
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
      Timer & T = GetTimer(T_CopyId); T.start();
      cudaMemcpy(h_u, d_u, nn * sizeof(double), cudaMemcpyDeviceToHost);
      T.stop();
      h_synchronized = true;
    }

    for (i=0; i<m_n_local[0]; i++) {
      for (j=0; j<m_n_local[1]; j++) {
        for (k=0; k<m_n_local[2]; k++) {
          f << " " << h_u[n2*k + n1*j + i];
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
  ::swap(d_u, other.d_u);
  ::swap(h_u, other.h_u);
  ::swap(h_synchronized, other.h_synchronized);
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

  if (!h_synchronized) {
    Timer & T = GetTimer(T_CopyId); T.start();
    cudaMemcpy(h_u, d_u, nn * sizeof(double), cudaMemcpyDeviceToHost);
    T.stop();
    h_synchronized = true;
  }

  Timer & T = GetTimer(T_OtherId); T.start();
  std::ostringstream s;
  int i, j, k;
  int imin = m_p.imin(0)-1;
  int jmin = m_p.imin(1)-1;
  int kmin = m_p.imin(2)-1;

  int imax = m_p.imax(0)+1;
  int jmax = m_p.imax(1)+1;
  int kmax = m_p.imax(2)+1;

  s << m_p.resultPath();
  mkdir_p(s.str().c_str());
  
  s << "/plot_" << std::setw(5) << std::setfill('0') << order << ".vtr";
  std::ofstream f(s.str().c_str());

  f << "<?xml version=\"1.0\"?>\n";
  f << "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
    << "<RectilinearGrid WholeExtent=\""
    << imin << " " << imax  << " " 
    << jmin << " " << jmax  << " " 
    << kmin << " " << kmax 
    << "\">\n"
    << "<Piece Extent=\""
    << imin << " " << imax  << " " 
    << jmin << " " << jmax  << " " 
    << kmin << " " << kmax 
    << "\">\n";

  f << "<PointData Scalars=\"values\">\n";
  f << "  <DataArray type=\"Float64\" Name=\"values\" format=\"ascii\">\n";
  
  for (k=kmin; k<=kmax; k++)
    for (j=jmin; j<=jmax; j++) {
      for (i=imin; i<=imax; i++)
        f << " " << operator()(i,j,k);
      f << "\n";
    }
  f << " </DataArray>\n";
   
  f << "</PointData>\n";

  f << " <Coordinates>\n";
  
  for (k=0; k<3; k++) {
    f << "   <DataArray type=\"Float64\" Name=\"" << char('X' + k) << "\"" 
      << " format=\"ascii\">";
    
    int imin = m_p.imin(k)-1;
    int imax = m_p.imax(k)+1;
    for (i=imin; i<=imax; i++)
      f << " " << i * m_p.dx(k);
    f << "   </DataArray>\n";
  }
  f << " </Coordinates>\n";
  
  f << "</Piece>\n"
    << "</RectilinearGrid>\n"
    << "</VTKFile>\n" <<std::endl;

  T.stop();
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
  h_synchronized = other.h_synchronized;

  Timer & T = GetTimer(T_CopyId); T.start();
  if (other.h_synchronized)
     memcpy(h_u, other.h_u, nn*sizeof(double));

  cudaMemcpy(d_u, other.d_u, nn * sizeof(double), cudaMemcpyDeviceToDevice);
  T.stop();
}