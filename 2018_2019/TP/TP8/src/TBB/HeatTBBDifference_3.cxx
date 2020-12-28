#include <cmath>
#include "Heat.hxx"

#include "tbb/tbb.h"


double Solver::Difference()
{  
  int n1 = m_u.n()-1;
  int m1 = m_u.m()-1;
  
  tbb::blocked_range2d<int, int> Indices(1, n1, n1/10,
                                         1, m1, m1/10);
  return tbb::parallel_reduce
    (Indices,
     0.0,
     
     [=](const tbb::blocked_range2d<int, int>& r, double diff)->double {
      int i,j;
      double local_diff = diff;
      for (i=r.rows().begin(); i<r.rows().end(); i++)
	for (j=r.cols().begin(); j<r.cols().end(); j++)
	  local_diff += std::abs(m_v(i,j) - m_u(i,j));
      
      return local_diff;
     },
     
     []( double x, double y )->double {
       return x+y;
     }
     );
}
