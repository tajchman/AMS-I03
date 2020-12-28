#include <cstdio>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include "Matrix.hxx"

#include "tbb/tbb.h"

class cMatrixCtor {
public:

  cMatrixCtor(Matrix & u, Matrix::fct f) : v(u), g(f)
  {
  }
    
  void operator() ( const tbb::blocked_range2d<int, int>& r ) const {
    
    int i,j;
    double x, y;
    for (i=r.rows().begin(); i<r.rows().end(); i++)
      for (j=r.cols().begin(); j<r.cols().end(); j++) {
	x = i*1.0/(v.n()-1);
	y = j*1.0/(v.m()-1);
	v(i,j) = g(x, y);
      }
  }
  
private:
  Matrix & v;
  Matrix::fct g;
};

Matrix::Matrix(int n, int m, Matrix::fct f_init, const char *name)
  : m_n(n), m_m(m), m_coefs(n*m), m_name(name)
{  
  cMatrixCtor MC(*this, f_init);
  tbb::blocked_range2d<int, int> 
    Indices(0, n, 10, 0, n, 10);
  tbb:parallel_for(Indices, MC);
}

void Matrix::save(int ksave) const
{
  int i,j;
  char s[1024];
  sprintf(s, "cpu_out_%04d.vtr", ksave);
  std::ofstream f(s);

  f << "<?xml version=\"1.0\"?>\n"
    << "<VTKFile type=\"RectilinearGrid\">\n"
    << "<RectilinearGrid WholeExtent=\"0 "
    << m_n-1 << " 0 " << m_m-1 << " 0 0\">\n"
    << "<Piece Extent=\"0 "
    << m_n-1 << " 0 " << m_m-1 << " 0 0\">\n";

  f << "<PointData Scalars=\"" << m_name <<"\">\n"
    << "<DataArray type=\"Float32\" Name=\"" << m_name 
    << "\" format=\"ascii\">\n";
  
  for (i=0; i<m_n; i++) {
    for (j=0; j<m_m; j++)
      f << " " << std::setw(12) << (*this)(i,j);
    f << "\n";
  }
  
  f << "</DataArray>\n"
    << "</PointData>\n";
    
  f << "<Coordinates>\n";
  f << "<DataArray type=\"Float32\" format=\"ascii\" "
    << "RangeMin=\"0\" RangeMax=\"" << m_n-1 << "\">";
  
  for (i=0; i<m_n; i++)
    f << " " << i*1.0/(m_n-1) ;
  f << "</DataArray>\n";
  f << "<DataArray type=\"Float32\" format=\"ascii\" "
    << "RangeMin=\"0\" RangeMax=\"" << m_m-1 << "\">";
  
  for (j=0; j<m_m; j++)
    f << " " << j*1.0/(m_m-1) ;
  f << "</DataArray>\n";
  
  f << "<DataArray type=\"Float32\" format=\"ascii\">\n0.0\n</DataArray>\n";
  f << "</Coordinates>\n";
  
  f << "</Piece>\n</RectilinearGrid>\n</VTKFile>\n";
  
}
