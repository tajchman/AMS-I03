#ifndef __VALUES__
#define __VALUES__

#include "parameters.hxx"
#include <vector>
#include <iostream>
#include <array>
class Values {

public:

  Values(Parameters & p);
  virtual ~Values() {}
  void operator= (const Values &);

  void init(callback_t f);
  void init();
  void boundaries(callback_t f);

  double & operator() (int i,int j,int k) {
    return m_u[n2*k + n1*j + i];
  }
  double operator() (int i,int j,int k) const {
    return m_u[n2*k + n1*j + i];
  }
  double & operator() (const std::array<int,3> & i) {
    return m_u[n2*i[2] + n1*i[1] + i[0]];
  }
  double operator() (const std::array<int,3> & i) const {
    return m_u[n2*i[2] + n1*i[1] + i[0]];
  }

  void plot(int order) const;
  void swap(Values & other);
  int size(int i) const { return m_imax[i] - m_imin[i] + 1; }
  void print(std::ostream &f) const;

private:

  Values(const Values &);
  int n1, n2;
  std::vector<double> m_u;
  Parameters & m_p;
  int m_imin[3];
  int m_imax[3];

  double m_dx[3];
  double m_xmin[3];
  double m_xmax[3];

};

std::ostream & operator<< (std::ostream & f, const Values & v);

#endif
