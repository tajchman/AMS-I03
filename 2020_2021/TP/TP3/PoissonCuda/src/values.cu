#include "values.hxx"
#include "os.hxx"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <cuda.h>

Values::Values(Parameters & prm) : m_p(prm)
{
  int i, nn = 1;
  for (i=0; i<3; i++)
    nn *= (m_n[i] = m_p.n(i));

  n1 = m_n[2];      // nombre de points dans la premiere direction
  n2 = m_n[1] * n1; // nombre de points dans le plan des 2 premieres directions
  
  cudaMalloc(&m_u, nn*sizeof(double));

  imin = m_p.imin(0);
  jmin = m_p.imin(1);
  kmin = m_p.imin(2);

  imax = m_p.imax(0);
  jmax = m_p.imax(1);
  kmax = m_p.imax(2);

  dx = m_p.dx(0);
  dy = m_p.dx(1);
  dz = m_p.dx(2);
  
  xmin =  m_p.xmin(0);
  ymin =  m_p.xmin(1);
  zmin =  m_p.xmin(2);
  xmax =  xmin + (imax-imin) * dx;
  ymax =  ymin + (jmax-jmin) * dy;
  zmax =  zmin + (kmax-kmin) * dz;
}

__global__
void zeroValue(int n, double *u)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i<n) {
    u[i] = 0.0;
  }
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
void initValue(int n0, int n1, int n2, 
               double xmin, double ymin, double zmin,
               double dx, double dy, double dz,
               double *u)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  int p;

  if (i<n0 && j<n1 && k<n2) {
    p = i + j*n0 + k*n0*n1;
    u[p] = cond_ini(xmin + i*dx, ymin + j*dy, zmin + k*dz);
  }
}

void Values::zero()
{
  int n = m_n[0]*m_n[1]*m_n[2];
  dim3 dimBlock(1024);
  dim3 dimGrid(ceil(n/double(dimBlock.x)));

  zeroValue<<<dimGrid, dimBlock>>>(n, m_u);
}

void Values::init()
{
  dim3 dimBlock(8,8,8);
  dim3 dimGrid(ceil(m_n[0]/double(dimBlock.x)),
               ceil(m_n[1]/double(dimBlock.y)),
               ceil(m_n[2]/double(dimBlock.z)));

  initValue<<<dimGrid, dimBlock>>>
        (m_n[0], m_n[1], m_n[2], 
         xmin,   ymin,   zmin, 
         dx,     dy,     dz, 
         m_u);
}

__global__
void boundZValue(int n0, int n1, int n2, int k, 
               double xmin, double ymin, double zmin,
               double dx, double dy, double dz,
               double *u)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int p;

  if (i<n0 && j<n1) {
    p = i + j*n0 + k*n0*n1;
    u[p] = cond_lim(xmin + i*dx, ymin + j*dy, zmin + k*dz);
  }
}

__global__
void boundYValue(int n0, int n1, int n2, int j,
               double xmin, double ymin, double zmin,
               double dx, double dy, double dz,
               double *u)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int k = threadIdx.y + blockIdx.y * blockDim.y;
  int p;

  if (i<n0 && k<n2) {
    p = i + j*n0 + k*n0*n1;
    u[p] = cond_lim(xmin + i*dx, ymin + j*dy, zmin + k*dz);
  }
}

__global__
void boundXValue(int n0, int n1, int n2, int i, 
               double xmin, double ymin, double zmin,
               double dx, double dy, double dz,
               double *u)
{
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int k = threadIdx.y + blockIdx.y * blockDim.y;
  int p;

  if (j<n1 && k<n2) {
    p = i + j*n0 + k*n0*n1;
    u[p] = cond_lim(xmin + i*dx, ymin + j*dy, zmin + k*dz);
  }
}

void Values::boundaries()
{
  dim3 dimBlock(16,16);
  dim3 dimGrid(ceil(m_n[0]/double(dimBlock.x)),
               ceil(m_n[1]/double(dimBlock.y)));

  boundZValue<<<dimGrid, dimBlock>>>
  (m_n[0], m_n[1], m_n[2], 0, 
   xmin,   ymin,   zmin, 
   dx,     dy,     dz, 
   m_u);

  boundZValue<<<dimGrid, dimBlock>>>
  (m_n[0], m_n[1], m_n[2], m_n[2]-1, 
   xmin,   ymin,   zmin, 
   dx,     dy,     dz, 
   m_u);

  boundYValue<<<dimGrid, dimBlock>>>
  (m_n[0], m_n[1], m_n[2], 0,
   xmin,   ymin,   zmin, 
   dx,     dy,     dz, 
   m_u);

  boundYValue<<<dimGrid, dimBlock>>>
  (m_n[0], m_n[1], m_n[2], m_n[1]-1,
   xmin,   ymin,   zmin, 
   dx,     dy,     dz, 
   m_u);

  boundXValue<<<dimGrid, dimBlock>>>
  (m_n[2], m_n[1], m_n[2], 0,
   xmin,   ymin,   zmin, 
   dx,     dy,     dz, 
   m_u);
  
  boundXValue<<<dimGrid, dimBlock>>>
  (m_n[0], m_n[1], m_n[2], m_n[0] - 1, 
   xmin,   ymin,   zmin, 
   dx,     dy,     dz, 
   m_u);
}


std::ostream & operator<< (std::ostream & f, const Values & v)
{
  v.print(f);
  return f;
}

void Values::print(std::ostream & f) const
{
    int i, j, k;
    int imin = m_p.imin(0);
    int jmin = m_p.imin(1);
    int kmin = m_p.imin(2);

    int imax = m_p.imax(0);
    int jmax = m_p.imax(1);
    int kmax = m_p.imax(2);

    for (i=imin; i<imax; i++) {
      for (j=jmin; j<jmax; j++) {
        for (k=kmin; k<kmax; k++)
          f << " " << operator()(i,j,k);
        f << std::endl;
        }
        f << std::endl;
      }
}

void Values::swap(Values & other)
{
  double * temp_u = m_u;
  m_u = other.m_u;
  other.m_u = temp_u;
  
  int i, temp;
  for (i=0; i<3; i++) {
    temp = m_n[i];
    m_n[i] = other.m_n[i];
    other.m_n[i] = temp;
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
        f << " " << operator()(i,j,k);
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
  
  for (i=0; i<3; i++)
    m_n[i] = other.m_n[i];
  
  m_u = other.m_u;
}
