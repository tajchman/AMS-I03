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
  int size(int i) const { return m_imax[i] - m_imin[i] + 1; }
  void print(std::ostream &f) const;

private:

  Values(const Values &);
  int n1, n2, nn;
  double * m_u;
  Parameters & m_p;
  int m_imin[3];
  int m_imax[3];
  int m_n_local[3];

  double m_dx[3];
  double m_xmin[3];
  double m_xmax[3];

};

std::ostream & operator<< (std::ostream & f, const Values & v);

#endif
