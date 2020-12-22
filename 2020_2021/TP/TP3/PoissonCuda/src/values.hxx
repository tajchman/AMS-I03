#ifndef __VALUES__
#define __VALUES__

#include "parameters.hxx"
#include <vector>
#include <iostream>

class Values {

public:

  Values(Parameters & p);
  virtual ~Values() {}
  void operator= (const Values &);
  
  void init();
  void zero();
  void boundaries();

  void plot(int order) const;
  void swap(Values & other);
  int size(int i) const { return m_n[i]; }
  void print(std::ostream &f) const;
  
private:
  
  Values(const Values &);
  int n1, n2;
  double * m_u;
  int m_n[3];
  Parameters & m_p;
  int imin, jmin, kmin;
  int imax, jmax, kmax;

  double dx, dy, dz;
  double xmin, ymin, zmin;
  double xmax, ymax, zmax;

};
				   
std::ostream & operator<< (std::ostream & f, const Values & v);

#endif
