#include "Heat.hxx"

#include "tbb/tbb.h"

void Solver::Iteration() {

 tbb::blocked_range2d<int, int> 
       Indices(1, m_u.n()-1, 40, 1, m_u.m()-1, 40);
  
 tbb::parallel_for
   (Indices,
    [=](const tbb::blocked_range2d<int, int>& r) {
     int i,j;
     for (i=r.rows().begin(); i<r.rows().end(); i++)
       for (j=r.cols().begin(); j<r.cols().end(); j++)
	 m_v(i,j) = m_u(i,j)
	   - m_lambda * (4*m_u(i,j)
			 - m_u(i+1,j) - m_u(i-1,j)
			 - m_u(i,j+1) - m_u(i,j-1))
	   + m_f(i,j) * m_dt;    
   });
 
 m_t += m_dt;
}

