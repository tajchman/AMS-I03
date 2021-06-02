#include "values.hxx"
#include "os.hxx"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

Values::Values(Parameters & prm) : m_p(prm)
{
  int i, nn = 1;

  for (i=0; i<3; i++) {
    m_imin[i] = m_p.imin(i);
    m_imax[i] = m_p.imax(i);
    m_dx[i] = m_p.dx(i);
    m_xmin[i] =  m_p.xmin(i);
    m_xmax[i] =  m_p.xmax(i);
    nn *= (m_imax[i] - m_imin[i] + 3);
  }
  
  n1 = m_imax[1] - m_imin[1] + 3;      // nombre de points dans la premiere direction
  n2 = (m_imax[2] - m_imin[2] + 3) * n1; // nombre de points dans le plan des 2 premieres directions
  m_u.resize(nn);
}

void Values::init()
{
  int i, j, k;
#ifdef _OPENMP
  int iThread = omp_get_thread_num();
#else
  int iThread = 0;
#endif

  int imin_thread = m_p.imin_local(0, iThread);
  int jmin_thread = m_p.imin_local(1, iThread);
  int kmin_thread = m_p.imin_local(2, iThread);

  int imax_thread = m_p.imax_local(0, iThread);
  int jmax_thread = m_p.imax_local(1, iThread);
  int kmax_thread = m_p.imax_local(2, iThread);

  double dx = m_p.dx(0), dy = m_p.dx(1), dz = m_p.dx(2);

  for (i=imin_thread; i<=imax_thread; i++)
    for (j=jmin_thread; j<=jmax_thread; j++)
      for (k=kmin_thread; k<=kmax_thread; k++)
        operator()(i,j,k) = 0.0;
}

void Values::init(callback_t f)
{
  int i, j, k;
#ifdef _OPENMP
  int iThread = omp_get_thread_num();
#else
  int iThread = 0;
#endif

  int imin_thread = m_p.imin_local(0, iThread);
  int jmin_thread = m_p.imin_local(1, iThread);
  int kmin_thread = m_p.imin_local(2, iThread);

  int imax_thread = m_p.imax_local(0, iThread);
  int jmax_thread = m_p.imax_local(1, iThread);
  int kmax_thread = m_p.imax_local(2, iThread);

  double dx = m_p.dx(0), dy = m_p.dx(1), dz = m_p.dx(2);

  for (i=imin_thread; i<=imax_thread; i++)
    for (j=jmin_thread; j<=jmax_thread; j++)
      for (k=kmin_thread; k<=kmax_thread; k++)
        operator()(i,j,k) = f(xmin + i*dx, ymin + j*dy, zmin + k*dz);
}

void Values::boundaries(callback_t f)
{
  int i[3];
  double x[3];
  for (int idim=0; idim<3; idim++) {

    int jdim = (idim+1)%3;
    int kdim = (idim+2)%3;

    int omin = m_imin[idim], omax = m_imax[idim];
    int pmin = m_imin[jdim], pmax = m_imax[jdim];
    int qmin = m_imin[kdim], qmax = m_imax[kdim];

    int p, q;

    i[idim] = omin-1;
    x[idim] = m_xmin[idim];
    for (p=pmin-1; p<=pmax+1; p++)
      for (q=qmin-1; q<=qmax+1; q++) {
        i[jdim] = p; i[kdim] = q;
        x[jdim] = m_xmin[jdim] + p*m_dx[jdim];
        x[kdim] = m_xmin[kdim] + q*m_dx[kdim];
        operator()(i[0], i[1], i[2]) = f(x[0], x[1], x[2]);
      }
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
  int imin = m_imin[0], jmin = m_imin[1], kmin = m_imin[2];
  int imax = m_imax[0], jmax = m_imax[1], kmax = m_imax[2];

  for (i=imin; i<=imax; i++) {
    for (j=jmin; j<=jmax; j++) {
      for (k=kmin; k<=kmax; k++)
        f << " " << operator()(i,j,k);
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
  m_u.swap(other.m_u);
  int i;
  for (i=0; i<3; i++) {
    ::swap(m_imin[i], other.m_imin[i]);
    ::swap(m_imax[i], other.m_imax[i]);
    ::swap(m_dx[i], other.m_dx[i]);
    ::swap(m_xmin[i], other.m_xmin[i]);
    ::swap(m_xmax[i], other.m_xmax[i]);
  }
  ::swap(n1, other.n1);
  ::swap(n2, other.n2);
}

void Values::plot(int order) const {

  std::ostringstream s;
  int i, j, k;
  int imin = m_imin[0]-1, jmin = m_imin[1]-1, kmin = m_imin[2]-1;
  int imax = m_imax[0]+1, jmax = m_imax[1]+1, kmax = m_imax[2]+1;

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
