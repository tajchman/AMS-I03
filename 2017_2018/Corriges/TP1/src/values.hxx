#ifndef __VALUES__
#define __VALUES__

#include "parameters.hxx"
#include <vector>

class Values {

public:

  Values(Parameters & p,
         double (*f)(double, double, double) = 0L);

  double & operator() (int i,int j,int k) {
    return m_u[i + m_n[0]*j + m_n[0]*m_n[1]*k];
  }
  double operator() (int i,int j,int k) const {
    return m_u[i + m_n[0]*j + m_n[0]*m_n[1]*k];
  }

  void plot(int order);
  void swap(Values & other);
  int size(int i) const { return m_n[i]; }
  
private:
  
  std::vector<double> m_u;
  int m_n[3];
  Parameters & m_p;
};
				   

#endif
