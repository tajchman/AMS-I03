#ifndef __PARTICULES__
#define __PARTICULES__

#include <stdlib.h>
#include <vector>

struct Particules {

  Particules(int m) : n(m) {

    x = new double[n];
    y = new double[n];
    z = new double[n];
    vx = new double[n];
    vy = new double[n];
    vz = new double[n];
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
    delete [] x;
    delete [] y;
    delete [] z;
    delete [] vx;
    delete [] vy;
    delete [] vz;
  }
  void move(double dT);

  double *x, *y, *z, *vx, *vy, *vz;
  int n;
};

#endif
