#ifndef __PARTICULES__
#define __PARTICULES__

#include <stdlib.h>
#include <vector>

struct Particule {

  Particule() {
    x = 2*float(rand())/RAND_MAX - 1;
    y = 2*float(rand())/RAND_MAX - 1;
    z = 2*float(rand())/RAND_MAX - 1;
    vx = 2*float(rand())/RAND_MAX - 1;
    vy = 2*float(rand())/RAND_MAX - 1;
    vz = 2*float(rand())/RAND_MAX - 1;
  }

  float x, y, z, vx, vy, vz;
};

struct Particules {

  Particules(int n) : p(n) {}
  void move(float dT);

  std::vector<Particule> p;
  float K;

};

#endif
