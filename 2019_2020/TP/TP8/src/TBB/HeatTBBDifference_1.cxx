#include <cmath>
#include "Heat.hxx"

#include "tbb/tbb.h"


class cDifference {
public:

  cDifference(const Matrix & uu,
	      const Matrix & vv)
    : u(uu), v(vv), diff(0.0) {}
  
  cDifference(cDifference& x, tbb::split )
    : u(x.u), v(x.v), diff(0.0) {}
  
  void join(cDifference & y ) {diff += y.diff; }

  void operator() ( const tbb::blocked_range2d<int, int>& r ) {
    
    int i,j;
    double local_diff = 0;
    for (i=r.rows().begin(); i<r.rows().end(); i++)
      for (j=r.cols().begin(); j<r.cols().end(); j++)
	local_diff += std::abs(v(i,j) - u(i,j));

    diff += local_diff;
  } 

  double result() const { return diff; }
  
private:
  const Matrix & v;
  const Matrix & u;
  double diff; 
};

double Solver::Difference()
{  
  cDifference Dif(m_u, m_v);
  int n1 = m_u.n()-1;
  int m1 = m_u.m()-1;
  
  tbb::blocked_range2d<int, int> Indices(1, n1, n1/10,
                                         1, m1, m1/10);
  tbb:parallel_reduce(Indices, Dif);

  return Dif.result();
}
