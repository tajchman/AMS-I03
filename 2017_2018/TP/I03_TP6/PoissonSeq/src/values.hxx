#ifndef __VALUES__
#define __VALUES__

#include "parameters.hxx"
#include <vector>

class Values {

public:

  Values() : m_p(NULL), n1(0), n2(0), m_u(NULL) {}
  virtual ~Values() {
	deallocate();
  };

  void init(const Parameters * p,
         double (*f)(double, double, double) = 0L);

  double & operator() (int i,int j,int k) {
    return m_u[n2*i + n1*j + k];
  }
  double operator() (int i,int j,int k) const {
    return m_u[n2*i + n1*j + k];
  }

  void plot(int order) const;
  void swap(Values & other);
  int size(int i) const { return m_n[i]; }
  
protected:
  
  int n1, n2;
  double * m_u;
  int m_n[3];
  const Parameters * m_p;

  virtual void allocate(size_t nn);
  virtual void deallocate();
};
				   

#endif
