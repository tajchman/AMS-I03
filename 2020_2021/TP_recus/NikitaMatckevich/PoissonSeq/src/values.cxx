#include "values.hxx"
#include "os.hxx"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>

Values::Values(const Parameters* prm) : m_p(prm), m_u(prm->n(0)*prm->n(1)*prm->n(2)) {}

Values::Values(Values&& other) : m_p(other.m_p), m_u(std::move(other.m_u)) {}

void Values::swap(Values& other) noexcept
{
  m_p = other.m_p;
  m_u.swap(other.m_u);
}

Values& Values::operator=(Values other) noexcept
{
  swap(other);
  return *this;
}

void Values::init()
{
  for (int i=imin(); i<imax(); i++)
    for (int j=jmin(); j<jmax(); j++)
      for (int k=kmin(); k<kmax(); k++)
        operator()(i,j,k) = 0.;
}

void Values::init(const callback_t& f)
{
  for (int i=imin(); i<imax(); i++)
    for (int j=jmin(); j<jmax(); j++)
      for (int k=kmin(); k<kmax(); k++)
        operator()(i,j,k) = f(xmin() + i*dx(), ymin() + j*dy(), zmin() + k*dz());
}

void Values::boundaries(const callback_t& f)
{
  int i, j, k;

  for (j=jmin(); j<jmax(); j++)
    for (k=kmin(); k<kmax(); k++) 
    {
      operator()(imin(),   j, k) = f(xmin(), ymin() + j*dy(), zmin() + k*dz());
      operator()(imax()-1, j, k) = f(xmax(), ymin() + j*dy(), zmin() + k*dz());
    }

  for (i=imin(); i<imax(); i++)
    for (k=kmin(); k<kmax(); k++)
    {
      operator()(i, jmin(),   k) = f(xmin()+ i*dx(), ymin(), zmin() + k*dz());
      operator()(i, jmax()-1, k) = f(xmin()+ i*dx(), ymax(), zmin() + k*dz());
    }

  for (i=imin(); i<imax(); i++)
    for (j=jmin(); j<jmax(); j++)
    {
      operator()(i, j, kmin()  ) = f(xmin()+ i*dx(), ymin() + j*dy(), zmax());
      operator()(i, j, kmax()-1) = f(xmin()+ i*dx(), ymin() + j*dy(), zmax());
    }
}

std::ostream& operator<<(std::ostream& f, const Values& v)
{
  v.print(f);
  return f;
}

void Values::print(std::ostream& f) const
{
  for (int i=imin(); i<imax(); i++) {
    for (int j=jmin(); j<jmax(); j++) {
      for (int k=kmin(); k<kmax(); k++)
        f << " " << operator()(i,j,k);
      f << std::endl;
    }
    f << std::endl;
  }
}

void Values::plot(int order) const {

  std::ostringstream s;
  
  s << m_p->resultPath();
  s << "/0";
  mkdir_p(s.str().c_str());
  
  s << "/plot_" << order << ".vtr";
  std::ofstream f(s.str().c_str());

  f << "<?xml version=\"1.0\"?>\n";
  f << "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
    << "<RectilinearGrid WholeExtent=\""
    << imin() << " " << imax()-1  << " " 
    << jmin() << " " << jmax()-1  << " " 
    << kmin() << " " << kmax()-1 
    << "\">\n"
    << "<Piece Extent=\""
    << imin() << " " << imax()-1  << " " 
    << jmin() << " " << jmax()-1  << " " 
    << kmin() << " " << kmax()-1 
    << "\">\n";

  f << "<PointData Scalars=\"values\">\n";
  f << "  <DataArray type=\"Float64\" Name=\"values\" format=\"ascii\">\n";
  
  for (int k=kmin(); k<kmax(); k++)
    for (int j=jmin(); j<jmax(); j++) {
      for (int i=imin(); i<imax(); i++)
        f << " " << operator()(i,j,k);
      f << "\n";
    }
  f << " </DataArray>\n";
   
  f << "</PointData>\n";

  f << " <Coordinates>\n";
  
  for (int k=0; k<3; k++) {
    f << "   <DataArray type=\"Float64\" Name=\"" << char('X' + k) << "\"" 
      << " format=\"ascii\">";
    for (int i=imin(); i<imax(); i++)
      f << " " << i * m_p->dx(k);
    f << "   </DataArray>\n";
  }
  f << " </Coordinates>\n";
  
  f << "</Piece>\n"
    << "</RectilinearGrid>\n"
    << "</VTKFile>\n" <<std::endl;
}
