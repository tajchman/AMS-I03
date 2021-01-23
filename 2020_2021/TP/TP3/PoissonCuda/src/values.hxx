#ifndef __VALUES__
#define __VALUES__

#include "parameters.hxx"
#include <vector>
#include <iostream>

class Values {

public:

  Values(Parameters & p);
  virtual ~Values();
  void operator= (const Values &);
  
  void init();
  void zero();
  void boundaries();

  double & operator() (int i,int j,int k) {
    return h_u[n2*k + n1*j + i];
  }
  double operator() (int i,int j,int k) const {
    return h_u[n2*k + n1*j + i];
  }
  
  void plot(int order) const;
  void swap(Values & other);
  int size(int i) const { return m_imax[i] - m_imin[i] + 1; }
  void print(std::ostream &f) const;

  double * dataCPU() { return m_u; }
  double * dataGPU() { return h_u; }
  
private:

  Values(const Values &);
  int n1, n2, nn;
  double * m_u, * h_u;
  mutable bool h_synchronized;

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
