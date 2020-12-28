#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include "calcul.h"
#include "cuda.h"
#include <cstdio>

void save(int ksave, const double *u, int n)
{
  int i,j;
  char s[1024];
  sprintf(s, "gpu_out_%d.vtr", ksave);

  std::vector<double> h_u(n*n);
  cudaMemcpy(h_u.data(), u, sizeof(double)*n*n, cudaMemcpyDeviceToHost);
  
  std::ofstream f(s);

  f << "<?xml version=\"1.0\"?>\n"
    << "<VTKFile type=\"RectilinearGrid\">\n"
    << "<RectilinearGrid WholeExtent=\"0 "
    << n-1 << " 0 " << n-1 << " 0 0\">\n"
    << "<Piece Extent=\"0 "
    << n-1 << " 0 " << n-1 << " 0 0\">\n";

  f << "<PointData Scalars=\"u\">\n"
    << "<DataArray type=\"Float32\" Name=\"u\" format=\"ascii\">\n";
  
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++)
      f << " " << h_u[i*n+j];
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
    << "RangeMin=\"0\" RangeMax=\"" << n-1 << "\">";
  
  for (i=0; i<n; i++)
    f << " " << i*1.0/n ;
  f << "</DataArray>\n";
  
  f << "<DataArray type=\"Float32\" format=\"ascii\">\n0.0\n</DataArray>\n";
  f << "</Coordinates>\n";
  
  f << "</Piece>\n</RectilinearGrid>\n</VTKFile>\n";

 
}
