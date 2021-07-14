#pragma once

#include <vector>
#include <iomanip>

class Values {
public:

  Values(int n, int m, 
         double xmin = 0, double xmax = 1,
         double ymin = 0, double ymax = 1) 
         : _n(n), _m(m), 
           _xmin(xmin), _xmax(xmax),
           _ymin(ymin), _ymax(ymax),
           _v((_n+2)*(_m+2))
  {    
  }

  inline double & operator()(int i, int j) {
    return _v[(_m+2)*i + j];
  }
  inline double operator()(int i, int j) const {
    return _v[(_m+2)*i + j];
  }

  int n() const { return _n; }
  int m() const { return _m; }
  double xmin() const { return _xmin; }
  double xmax() const { return _xmax; }
  double ymin() const { return _ymin; }
  double ymax() const { return _ymax; }
  double dx() const { return (_xmax - _xmin) / (_n+1); }
  double dy() const { return (_ymax - _ymin) / (_m+1); }

protected:

  int _n, _m;
  double _xmin, _ymin, _xmax, _ymax;
  std::vector<double> _v;

  friend void swap(Values &u, Values &v);
};

inline void swap(Values &u, Values &v)
{
  u._v.swap(v._v);
}

std::ostream & operator<< (std::ostream & f, Values & v) {
  int i, n = v.n(), j, m = v.m();
  for (i=0; i<n+2; i++) {
    f << std::setw(6) << i << ":";
    for (j=0; j<m+2;j++)
       f << " " << std::setw(9) << std::setprecision(3) << v(i,j);
    f << std::endl; 
  }
  return f;
}