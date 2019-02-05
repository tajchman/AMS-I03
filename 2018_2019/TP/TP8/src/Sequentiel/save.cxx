#include <iostream>
#include <fstream>
#include <string>
#include "Matrix.hxx"
#include <cstdio>

void save(int ksave, const Matrix & M)
{
  int i,j, n = M.n(), m = M.m();
  char s[1024];
  sprintf(s, "cpu_out_%d.vtr", ksave);
  std::ofstream f(s);

  f << "<?xml version=\"1.0\"?>\n"
    << "<VTKFile type=\"RectilinearGrid\">\n"
    << "<RectilinearGrid WholeExtent=\"0 "
    << n-1 << " 0 " << m-1 << " 0 0\">\n"
    << "<Piece Extent=\"0 "
    << n-1 << " 0 " << m-1 << " 0 0\">\n";

  f << "<PointData Scalars=\"u\">\n"
    << "<DataArray type=\"Float32\" Name=\"u\" format=\"ascii\">\n";
  
  for (i=0; i<n; i++) {
    for (j=0; j<m; j++)
      f << " " << M(i,j);
    f << "\n";
  }
  
  f << "</DataArray>\n"
    << "</PointData>\n";
    
  f << "<Coordinates>\n";
  f << "<DataArray type=\"Float32\" format=\"ascii\" "
    << "RangeMin=\"0\" RangeMax=\"" << n-1 << "\">";
  
  for (i=0; i<n; i++)
    f << " " << i*1.0/n ;
  f << "</DataArray>\n";
  f << "<DataArray type=\"Float32\" format=\"ascii\" "
    << "RangeMin=\"0\" RangeMax=\"" << m-1 << "\">";
  
  for (j=0; j<m; j++)
    f << " " << j*1.0/m ;
  f << "</DataArray>\n";
  
  f << "<DataArray type=\"Float32\" format=\"ascii\">\n0.0\n</DataArray>\n";
  f << "</Coordinates>\n";
  
  f << "</Piece>\n</RectilinearGrid>\n</VTKFile>\n";
  
}
