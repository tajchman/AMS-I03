#ifndef __PARTICULES__
#define __PARTICULES__

#include <stdlib.h>
#include <xmmintrin.h>

#define alignment 8

struct Particules {

  Particules(int m) : n(m) {

    x = (double *) _mm_malloc(n*sizeof(double),alignment);
    y = (double *) _mm_malloc(n*sizeof(double),alignment);
    z = (double *) _mm_malloc(n*sizeof(double),alignment);
    vx = (double *) _mm_malloc(n*sizeof(double),alignment);
    vy = (double *) _mm_malloc(n*sizeof(double),alignment);
    vz = (double *) _mm_malloc(n*sizeof(double),alignment);

    for (int i=0; i<n; i++) {
      x[i]  = 2*double(rand())/RAND_MAX - 1;
      y[i]  = 2*double(rand())/RAND_MAX - 1;
      z[i]  = 2*double(rand())/RAND_MAX - 1;
      vx[i] = 2*double(rand())/RAND_MAX - 1;
      vy[i] = 2*double(rand())/RAND_MAX - 1;
      vz[i] = 2*double(rand())/RAND_MAX - 1;
    }
  }
  ~Particules() {
    _mm_free(x);
    _mm_free(y);
    _mm_free(z);
    _mm_free(vx);
    _mm_free(vy);
    _mm_free(vz);
  }
  void move(double dT);

  double *x, 
         *y, 
         *z, 
         *vx, 
         *vy, 
         *vz;
  int n;
  double K;
};

#endif
