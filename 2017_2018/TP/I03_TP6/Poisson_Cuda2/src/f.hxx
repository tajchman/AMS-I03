/*
 * f.hpp
 *
 *  Created on: 13 févr. 2018
 *      Author: marc
 */

#ifndef F_HPP_
#define F_HPP_

#ifdef __CUDACC__
__device__
double f_GPU(double x, double y, double z)
#else
double f_CPU(double x, double y, double z)
#endif
{
  x -= 0.5;
  y -= 0.5;
  z -= 0.5;
  double f = (x*x+y*y+z*z < 0.1) ? 1.0 : 0.0;
  return f;
}



#endif /* F_HPP_ */
