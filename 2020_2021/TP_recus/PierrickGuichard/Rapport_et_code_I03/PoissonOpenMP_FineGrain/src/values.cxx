#include "values.hxx"
#include "os.hxx"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>

Values::Values(Parameters & prm) : m_p(prm)
{
  int i, nn = 1;
  for (i=0; i<3; i++)
    nn *= (m_n[i] = m_p.n(i));

  n1 = m_n[2];      // nombre de points dans la premiere direction
  n2 = m_n[1] * n1; // nombre de points dans le plan des 2 premieres directions
  
  m_u.resize(nn);

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

void Values::init()
{
  int i, j, k;
  # pragma omp parallel for private(i, j, k)
  for (i=imin; i<imax; i++)
    for (j=jmin; j<jmax; j++)
      for (k=kmin; k<kmax; k++)
        operator()(i,j,k) = 0.0;
}

void Values::init(callback_t f)
{
  int i, j, k;
  # pragma omp parallel for private (i, j, k)
  for (i=imin; i<imax; i++)
    for (j=jmin; j<jmax; j++)
      for (k=kmin; k<kmax; k++)
        operator()(i,j,k) = f(xmin + i*dx, ymin + j*dy, zmin + k*dz);
}

void Values::boundaries(callback_t f)
{
  int i, j, k;
  # pragma omp parallel for private (j, k)
  for (j=jmin; j<jmax; j++)
    for (k=kmin; k<kmax; k++) 
    {
      operator()(imin,   j, k) = f(xmin, ymin + j*dy, zmin + k*dz);
      operator()(imax-1, j, k) = f(xmax, ymin + j*dy, zmin + k*dz);
    }
  # pragma omp parallel for private (i, k)
  for (i=imin; i<imax; i++)
    for (k=kmin; k<kmax; k++)
    {
      operator()(i, jmin,   k) = f(xmin+ i*dx, ymin, zmin + k*dz);
      operator()(i, jmax-1, k) = f(xmin+ i*dx, ymax, zmin + k*dz);
    }
  # pragma omp parallel for private (i, j)
  for (i=imin; i<imax; i++)
    for (j=jmin; j<jmax; j++)
    {
      operator()(i, j, kmin  ) = f(xmin+ i*dx, ymin + j*dy, zmax);
      operator()(i, j, kmax-1) = f(xmin+ i*dx, ymin + j*dy, zmax);
    }
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

    # pragma omp parallel for private(i, j, k)
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
  m_u.swap(other.m_u);
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
#ifdef _OPENMP
  s << "/" << m_p.nthreads();
#else
  s << "/0";
#endif
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
