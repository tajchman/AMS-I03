#ifndef __VALUES__
#define __VALUES__

#include "parameters.hxx"
#include <vector>
#include <iostream>

class Values {

public:

  Values(const Parameters * p);
  virtual ~Values() {}
  void operator= (const Values &);
  
  void init(double (*f)(double, double, double) = 0L);

  double & operator() (size_t i,size_t j,size_t k) {
    return m_u[n2*i + n1*j + k];
  }
  double operator() (size_t i,size_t j,size_t k) const {
    return m_u[n2*i + n1*j + k];
  }

  void plot(int order) const;
  void swap(Values & other);
  size_t size(int i) const { return m_n[i]; }
  void print(std::ostream &f) const;
  
private:
  
  Values(const Values &);
  size_t n1, n2;
  std::vector<double> m_u;
  size_t m_n[3];
  const Parameters * m_p;
};
				   

#endif
