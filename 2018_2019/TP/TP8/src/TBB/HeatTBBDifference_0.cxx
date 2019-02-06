#include "Heat.hxx"

#include "tbb/tbb.h"


class cDifference {
public:

  cDifference(const Matrix & uu,
	      const Matrix & vv)
    : u(uu), v(vv), diff(0.0) {}
  
  void operator() ( const tbb::blocked_range2d<int, int>& r ) {
    
    int i,j;
    diff = 0.0;
    for (i=r.rows().begin(); i<r.rows().end(); i++)
      for (j=r.cols().begin(); j<r.cols().end(); j++)
	diff += std::abs(v(i,j) - u(i,j));
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
  tbb::blocked_range2d<int, int> Indices(1, m_u.n()-1, 40,
                                         1, m_u.m()-1, 40);
  
  Dif(Indices);

  return Dif.result();
}
