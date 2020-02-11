#include "values.hxx"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>

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
  int imin = m_p->imin(0);
  int jmin = m_p->imin(1);
  int kmin = m_p->imin(2);

  int imax = m_p->imax(0);
  int jmax = m_p->imax(1);
  int kmax = m_p->imax(2);

  if (f) {
    double dx = m_p->dx(0), dy = m_p->dx(1), dz = m_p->dx(2);
    double xmin =  m_p->xmin(0);
    double ymin =  m_p->xmin(1);
    double zmin =  m_p->xmin(2);

#pragma omp parallel for default(shared) private(i,j,k)
    for (i=imin; i<imax; i++)
      for (j=jmin; j<jmax; j++)
        for (k=kmin; k<kmax; k++)
          operator()(i,j,k) = f(xmin + i*dx, ymin + j*dy, zmin + k*dz);
  }
  else {
#pragma omp parallel for default(shared) private(i,j,k)
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

void Values::synchronize() {

  int i, j, k, l, n = m_n[0], m = m_n[1], p = m_n[2];

  MPI_Status status;

  if (m_p->neighbor(0, 0) != MPI_PROC_NULL) {
    std::vector<double> buf_in(m * p);
    std::vector<double> buf_out(m * p);
    for (l = 0, k = 0; k < p; k++)
      for (j = 0; j < m; j++)
        buf_in[l++] = (*this)(1, j, k);

    MPI_Sendrecv(buf_in.data(), m * p, MPI_DOUBLE, m_p->neighbor(0, 0), 0,
        buf_out.data(), m * p, MPI_DOUBLE, m_p->neighbor(0, 0), 0, m_p->comm(),
        &status);

    for (l = 0, k = 0; k < p; k++)
      for (j = 0; j < m; j++) {
        (*this)(0, j, k) = buf_out[l++];
      }
  }

  if (m_p->neighbor(0, 1) != MPI_PROC_NULL) {
    std::vector<double> buf_in(m * p);
    std::vector<double> buf_out(m * p);
    for (l = 0, k = 0; k < p; k++)
      for (j = 0; j < m; j++) {
        buf_in[l++] = (*this)(n - 2, j, k);
      }

    MPI_Sendrecv(buf_in.data(), m * p, MPI_DOUBLE, m_p->neighbor(0, 1), 0,
        buf_out.data(), m * p, MPI_DOUBLE, m_p->neighbor(0, 1), 0, m_p->comm(),
        &status);

    for (l = 0, k = 0; k < p; k++)
      for (j = 0; j < m; j++)
        (*this)(n - 1, j, k) = buf_out[l++];
  }

  if (m_p->neighbor(1, 0) != MPI_PROC_NULL) {
    std::vector<double> buf_in(n * p);
    std::vector<double> buf_out(n * p);
    for (l = 0, k = 0; k < p; k++)
      for (i = 0; i < n; i++)
        buf_in[l++] = (*this)(i, 1, k);

    MPI_Sendrecv(buf_in.data(), n * p, MPI_DOUBLE, m_p->neighbor(1, 0), 0,
        buf_out.data(), n * p, MPI_DOUBLE, m_p->neighbor(1, 0), 0, m_p->comm(),
        &status);

    for (l = 0, k = 0; k < p; k++)
      for (i = 0; i < n; i++)
        (*this)(i, 0, k) = buf_out[l++];
  }

  if (m_p->neighbor(1, 1) != MPI_PROC_NULL) {
    std::vector<double> buf_in(n * p);
    std::vector<double> buf_out(n * p);
    for (l = 0, k = 0; k < p; k++)
      for (i = 0; i < n; i++)
        buf_in[l++] = (*this)(i, m - 2, k);

    MPI_Sendrecv(buf_in.data(), n * p, MPI_DOUBLE, m_p->neighbor(1, 1), 0,
        buf_out.data(), n * p, MPI_DOUBLE, m_p->neighbor(1, 1), 0, m_p->comm(),
        &status);

    for (l = 0, k = 0; k < p; k++)
      for (i = 0; i < n; i++)
        (*this)(i, m - 1, k) = buf_out[l++];
  }

  if (m_p->neighbor(2, 0) != MPI_PROC_NULL) {
    std::vector<double> buf_in(n * m);
    std::vector<double> buf_out(n * m);
    for (l = 0, j = 0; j < m; j++)
      for (i = 0; i < n; i++)
        buf_in[l++] = (*this)(i, j, 1);

    MPI_Sendrecv(buf_in.data(), n * m, MPI_DOUBLE, m_p->neighbor(2, 0), 0,
        buf_out.data(), n * m, MPI_DOUBLE, m_p->neighbor(2, 0), 0, m_p->comm(),
        &status);

    for (l = 0, j = 0; j < m; j++)
      for (i = 0; i < n; i++)
        (*this)(i, j, 0) = buf_out[l++];
  }

  if (m_p->neighbor(2, 1) != MPI_PROC_NULL) {
    std::vector<double> buf_in(n * m);
    std::vector<double> buf_out(n * m);
    for (l = 0, j = 0; j < m; j++)
      for (i = 0; i < n; i++)
        buf_in[l++] = (*this)(i, j, p - 2);

    MPI_Sendrecv(buf_in.data(), n * m, MPI_DOUBLE, m_p->neighbor(2, 1), 0,
        buf_out.data(), n * m, MPI_DOUBLE, m_p->neighbor(2, 1), 0, m_p->comm(),
        &status);

    for (l = 0, j = 0; j < m; j++)
      for (i = 0; i < n; i++)
        (*this)(i, j, p - 1) = buf_out[l++];
  }
}

