#ifndef __PARTICULES__
#define __PARTICULES__

#include <stdlib.h>
#include <vector>

struct Particule {

  Particule() {
    x = 2*double(rand())/RAND_MAX - 1;
    y = 2*double(rand())/RAND_MAX - 1;
    z = 2*double(rand())/RAND_MAX - 1;
    vx = 2*double(rand())/RAND_MAX - 1;
    vy = 2*double(rand())/RAND_MAX - 1;
    vz = 2*double(rand())/RAND_MAX - 1;
  }

  double x, y, z, vx, vy, vz;
};

struct Particules {

  Particules(int n) : p(n) {}
  void move(double dT);

  std::vector<Particule> p;

};

#endif
