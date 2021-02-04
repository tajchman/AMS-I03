#ifndef __VALUES__
#define __VALUES__

#include "parameters.hxx"
#include <vector>

class Values {

public:

  Values(const Parameters * p,
         double (*f)(double, double, double) = 0L);
  Values(const Values & v);

  void operator=(const Values & other);

  double & operator() (int i,int j,int k) {
    return m_u[n2*i + n1*j + k];
  }
  double operator() (int i,int j,int k) const {
    return m_u[n2*i + n1*j + k];
  }

  void plot(int order) const;
  void swap(Values & other);
  int size(int i) const { return m_n[i]; }
  void synchronize();


private:
  
  int n1, n2;
  std::vector<double> m_u;
  int m_n[3];
  double m_dx[3];
  double m_xmin[3];
  const Parameters * m_P;
};
				   

#endif
