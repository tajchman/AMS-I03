#ifndef __PARTICULES__
#define __PARTICULES__

#include <stdlib.h>
#include <vector>


struct Particules {

  Particules(int n) : x(n), y(n), z(n), vx(n), vy(n), vz(n) {

    for (int i=0; i<n; i++) {
    x[i]  = 2*double(rand())/RAND_MAX - 1;
    y[i]  = 2*double(rand())/RAND_MAX - 1;
    z[i]  = 2*double(rand())/RAND_MAX - 1;
    vx[i] = 2*double(rand())/RAND_MAX - 1;
    vy[i] = 2*double(rand())/RAND_MAX - 1;
    vz[i] = 2*double(rand())/RAND_MAX - 1;
    }
  }
  void move(double dT);

  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<double> vx;
  std::vector<double> vy;
  std::vector<double> vz;

};

#endif
