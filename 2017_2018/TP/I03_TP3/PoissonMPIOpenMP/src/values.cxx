#include "values.hxx"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>

Values::Values(const Parameters * prm,
               double (*f)(double, double, double)) : m_P(prm)
{
  int i, nn = 1;
  int n = m_P->n(0), m = m_P->n(1), p = m_P->n(2);
  for (i=0; i<3; i++)
    nn *= (m_n[i] = m_P->n(i));

  n1 = m_n[2];      // nombre de points dans la premiere direction
  n2 = m_n[1] * n1; // nombre de points dans le plan des 2 premieres directions

  if (f) {
    m_u.resize(nn);

    int j, k;

    for (i=0; i<3; i++) {
	m_dx[i] = m_P->dx(i);
	m_xmin[i] = m_P->xmin(i);
    }

    double xmin = m_xmin[0],
      ymin =  m_xmin[1],
      zmin =  m_xmin[2];

    for (i=0; i<m_n[0]; i++)
      for (j=0; j<m_n[1]; j++)
        for (k=0; k<m_n[2]; k++)
          operator()(i,j,k) = f(xmin + i*m_dx[0],
                                ymin + j*m_dx[1],
                                zmin + k*m_dx[2]);
  }
  else
    m_u.assign(nn, 0.0);
}

void Values::operator=(const Values & other) {
  m_u = other.m_u;

  int i;
  for (i=0; i<3; i++) {
    m_n[i] = other.m_n[i];
    m_dx[i] = other.m_dx[i];
    m_xmin[i] = other.m_xmin[i];
  }
  n1 = other.n1;
  n2 = other.n2;
  m_P = other.m_P;
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
  s << m_P->resultPath() << "/plot_"
    << order << ".vtr";
  std::ofstream f(s.str().c_str());

  f << "<?xml version=\"1.0\"?>\n";
  f << "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
    << "<RectilinearGrid WholeExtent=\""
    << 0 << " " << m_P->n(0) - 1  << " "
    << 0 << " " << m_P->n(1) - 1  << " "
    << 0 << " " << m_P->n(2) - 1
    << "\">\n"
    << "<Piece Extent=\""
    << 0 << " " << m_P->n(0) - 1 << " "
    << 0 << " " << m_P->n(1) - 1 << " "
    << 0 << " " << m_P->n(2) - 1
    << "\">\n";

  f << "<PointData Scalars=\"values\">\n";
  f << "  <DataArray type=\"Float64\" Name=\"values\" format=\"ascii\">\n";
  
  for (k=0; k< m_P->n(2); k++) {
    for (j=0; j< m_P->n(1); j++) {
      for (i=0; i< m_P->n(0); i++) {
        f << " " << operator()(i,j,k);
      }
      f << "\n";
    }
  }
  f << " </DataArray>\n";
   
  f << "</PointData>\n";

  f << " <Coordinates>\n";
  
  for (k=0; k<3; k++) {
    f << "   <DataArray type=\"Float64\" Name=\"" << char('X' + k) << "\"" 
      << " format=\"ascii\">";
    
    for (i=0; i<m_P->n(k); i++)
      f << " " << i * m_P->dx(k);
    f << "   </DataArray>\n";
  }
  f << " </Coordinates>\n";
  
  f << "</Piece>\n"
    << "</RectilinearGrid>\n"
    << "</VTKFile>\n" <<std::endl;
}

#include <mpi.h>

void Values::synchronize() {

  int i, j, k, l, n = m_n[0], m = m_n[1], p = m_n[2];

  MPI_Status status;

  if (m_P->neighbor(0, 0) != MPI_PROC_NULL) {
    std::vector<double> buf_in(m * p);
    std::vector<double> buf_out(m * p);
    for (l = 0, k = 0; k < p; k++)
      for (j = 0; j < m; j++)
        buf_in[l++] = (*this)(1, j, k);

    MPI_Sendrecv(buf_in.data(), m * p, MPI_DOUBLE, m_P->neighbor(0, 0), 0,
        buf_out.data(), m * p, MPI_DOUBLE, m_P->neighbor(0, 0), 0, m_P->comm(),
        &status);

    for (l = 0, k = 0; k < p; k++)
      for (j = 0; j < m; j++) {
        (*this)(0, j, k) = buf_out[l++];
      }
  }

  if (m_P->neighbor(0, 1) != MPI_PROC_NULL) {
    std::vector<double> buf_in(m * p);
    std::vector<double> buf_out(m * p);
    for (l = 0, k = 0; k < p; k++)
      for (j = 0; j < m; j++) {
        buf_in[l++] = (*this)(n - 2, j, k);
      }

    MPI_Sendrecv(buf_in.data(), m * p, MPI_DOUBLE, m_P->neighbor(0, 1), 0,
        buf_out.data(), m * p, MPI_DOUBLE, m_P->neighbor(0, 1), 0, m_P->comm(),
        &status);

    for (l = 0, k = 0; k < p; k++)
      for (j = 0; j < m; j++)
        (*this)(n - 1, j, k) = buf_out[l++];
  }

  if (m_P->neighbor(1, 0) != MPI_PROC_NULL) {
    std::vector<double> buf_in(n * p);
    std::vector<double> buf_out(n * p);
    for (l = 0, k = 0; k < p; k++)
      for (i = 0; i < n; i++)
        buf_in[l++] = (*this)(i, 1, k);

    MPI_Sendrecv(buf_in.data(), n * p, MPI_DOUBLE, m_P->neighbor(1, 0), 0,
        buf_out.data(), n * p, MPI_DOUBLE, m_P->neighbor(1, 0), 0, m_P->comm(),
        &status);

    for (l = 0, k = 0; k < p; k++)
      for (i = 0; i < n; i++)
        (*this)(i, 0, k) = buf_out[l++];
  }

  if (m_P->neighbor(1, 1) != MPI_PROC_NULL) {
    std::vector<double> buf_in(n * p);
    std::vector<double> buf_out(n * p);
    for (l = 0, k = 0; k < p; k++)
      for (i = 0; i < n; i++)
        buf_in[l++] = (*this)(i, m - 2, k);

    MPI_Sendrecv(buf_in.data(), n * p, MPI_DOUBLE, m_P->neighbor(1, 1), 0,
        buf_out.data(), n * p, MPI_DOUBLE, m_P->neighbor(1, 1), 0, m_P->comm(),
        &status);

    for (l = 0, k = 0; k < p; k++)
      for (i = 0; i < n; i++)
        (*this)(i, m - 1, k) = buf_out[l++];
  }

  if (m_P->neighbor(2, 0) != MPI_PROC_NULL) {
    std::vector<double> buf_in(n * m);
    std::vector<double> buf_out(n * m);
    for (l = 0, j = 0; j < m; j++)
      for (i = 0; i < n; i++)
        buf_in[l++] = (*this)(i, j, 1);

    MPI_Sendrecv(buf_in.data(), n * m, MPI_DOUBLE, m_P->neighbor(2, 0), 0,
        buf_out.data(), n * m, MPI_DOUBLE, m_P->neighbor(2, 0), 0, m_P->comm(),
        &status);

    for (l = 0, j = 0; j < m; j++)
      for (i = 0; i < n; i++)
        (*this)(i, j, 0) = buf_out[l++];
  }

  if (m_P->neighbor(2, 1) != MPI_PROC_NULL) {
    std::vector<double> buf_in(n * m);
    std::vector<double> buf_out(n * m);
    for (l = 0, j = 0; j < m; j++)
      for (i = 0; i < n; i++)
        buf_in[l++] = (*this)(i, j, p - 2);

    MPI_Sendrecv(buf_in.data(), n * m, MPI_DOUBLE, m_P->neighbor(2, 1), 0,
        buf_out.data(), n * m, MPI_DOUBLE, m_P->neighbor(2, 1), 0, m_P->comm(),
        &status);

    for (l = 0, j = 0; j < m; j++)
      for (i = 0; i < n; i++)
        (*this)(i, j, p - 1) = buf_out[l++];
  }
}


