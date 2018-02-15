#include "CpuValues.hxx"
#include "f.hxx"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>

CpuValues::CpuValues(const CpuParameters * prm) : AbstractValues(prm)
{
  m_u = NULL;
  CpuValues::allocate(nn);
}

CpuValues::CpuValues(const CpuValues & other) : AbstractValues(other)
{
  m_u = NULL;
  CpuValues::allocate(other.nn);
  memcpy(m_u, other.m_u, sizeof(double)*nn);
}

void CpuValues::allocate(size_t n)
{
  deallocate();
  m_u = new double [n];
}

void CpuValues::deallocate()
{
  if (m_u == NULL) {
    delete [] m_u;
    m_u = NULL;
  }
}

void CpuValues::init()
{
  int i, j, k;
  int imin = m_p->imin(0);
  int jmin = m_p->imin(1);
  int kmin = m_p->imin(2);

  int imax = m_p->imax(0);
  int jmax = m_p->imax(1);
  int kmax = m_p->imax(2);

  for (i=imin; i<imax; i++)
    for (j=jmin; j<jmax; j++)
      for (k=kmin; k<kmax; k++)
	operator()(i,j,k) = 0.0;
}

void CpuValues::init_f()
{
  int i, j, k;
  int imin = m_p->imin(0);
  int jmin = m_p->imin(1);
  int kmin = m_p->imin(2);

  int imax = m_p->imax(0);
  int jmax = m_p->imax(1);
  int kmax = m_p->imax(2);

  double dx = m_p->dx(0), dy = m_p->dx(1), dz = m_p->dx(2);
  double xmin =  m_p->xmin(0);
  double ymin =  m_p->xmin(1);
  double zmin =  m_p->xmin(2);

  for (i=imin; i<imax; i++)
    for (j=jmin; j<jmax; j++)
      for (k=kmin; k<kmax; k++)
	operator()(i,j,k) = f_CPU(xmin + i*dx, ymin + j*dy, zmin + k*dz);
}

void CpuValues::operator= (const CpuValues &other)
{
  int i;
  size_t nn = 1;

  for (i=0; i<3; i++)
    nn *= (m_n[i] = other.m_n[i]);

  allocate(nn);
  memcpy(m_u, other.m_u, nn*sizeof(double));
}

void CpuValues::print(std::ostream & f) const
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

void CpuValues::plot(const char *prefix, int order) const {

  std::ostringstream s;
  int i, j, k;
  int imin = m_p->imin(0)-1;
  int jmin = m_p->imin(1)-1;
  int kmin = m_p->imin(2)-1;

  int imax = m_p->imax(0)+1;
  int jmax = m_p->imax(1)+1;
  int kmax = m_p->imax(2)+1;

  s << m_p->resultPath() << "plot_" << prefix
    << order << ".vtr";
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
    << jmin << " " << jmax-1 << " "
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

    int imin = m_p->imin(k)-1;
    int imax = m_p->imax(k)+1;
    for (i=imin; i<imax; i++)
      f << " " << i * m_p->dx(k);
    f << "   </DataArray>\n";
  }
  f << " </Coordinates>\n";

  f << "</Piece>\n"
    << "</RectilinearGrid>\n"
    << "</VTKFile>\n" <<std::endl;
}

