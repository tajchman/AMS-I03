#ifndef _CALCUL_HXX
#define _CALCUL_HXX

#include "OpenCL.hxx"

class Calcul_OpenCL {
public:
  Calcul_OpenCL(int n);
  ~Calcul_OpenCL();

  void init();
  void addition();
  double verification();
  
private:
  OpenCL CL;
  cl_kernel initKernel, addKernel;
  int n;
  cl_mem d_u, d_v, d_w;
  unsigned int blockSize, gridSize;
};

#endif

