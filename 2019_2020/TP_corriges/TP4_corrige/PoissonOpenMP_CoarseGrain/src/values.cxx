#include "values.hxx"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#if defined(_OPENMP)
   #include <omp.h>
#endif

Values::Values(const Parameters * prm)
{
  m_p = prm;
  int i, nn = 1;
  for (i=0; i<3; i++)
    nn *= (m_n[i] = m_p->n(i));

  n1 = m_n[2];      // nombre de points dans la premiere direction
  n2 = m_n[1] * n1; // nombre de points dans le plan des 2 premieres directions
  
  m_u.resize(nn);
}

void Values::init(double (*f)(double, double, double))
{
  int i, j, k;
  int ith = omp_get_thread_num();

  int imin = m_p->thread_imin(0, ith) ;
  int jmin = m_p->thread_imin(1, ith) ;
  int kmin = m_p->thread_imin(2, ith) ;

  int imax = m_p->thread_imax(0, ith) ;
  int jmax = m_p->thread_imax(1, ith) ;
  int kmax = m_p->thread_imax(2, ith) ;

  if (f) {
    double dx = m_p->dx(0), dy = m_p->dx(1), dz = m_p->dx(2);
    double xmin =  m_p->xmin(0);
    double ymin =  m_p->xmin(1);
    double zmin =  m_p->xmin(2);

    for (i=imin; i<imax; i++)
      for (j=jmin; j<jmax; j++)
        for (k=kmin; k<kmax; k++)
          operator()(i,j,k) = f(xmin + i*dx, ymin + j*dy, zmin + k*dz);
  }
  else {
    for (i=imin; i<imax; i++)
      for (j=jmin; j<jmax; j++)
        for (k=kmin; k<kmax; k++)
          operator()(i,j,k) = 0.0;

  }
}

void Values::print(std::ostream & f) const
{
    int i, j, k;
    int imin = m_p->imin(0);
    int jmin = m_p->imin(1);
    int kmin = m_p->imin(2);

    int imax = m_p->imax(0);
    int jmax = m_p->imax(1);
    int kmax = m_p->imax(2);

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
  int imin = m_p->imin(0);
  int jmin = m_p->imin(1);
  int kmin = m_p->imin(2);

  int imax = m_p->imax(0);
  int jmax = m_p->imax(1);
  int kmax = m_p->imax(2);

  s << m_p->resultPath() << "plot_"
    << order << ".vtr";
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
    
    int imin = m_p->imin(k);
    int imax = m_p->imax(k);
    for (i=imin; i<imax; i++)
      f << " " << i * m_p->dx(k);
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
