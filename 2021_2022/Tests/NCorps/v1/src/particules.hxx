#ifndef __PARTICULES__
#define __PARTICULES__

#include "reel.h"
#include <stdlib.h>
#include <vector>

struct Particule {

  Particule() {
    x = 2*reel(rand())/RAND_MAX - 1;
    y = 2*reel(rand())/RAND_MAX - 1;
    z = 2*reel(rand())/RAND_MAX - 1;
    vx = 2*reel(rand())/RAND_MAX - 1;
    vy = 2*reel(rand())/RAND_MAX - 1;
    vz = 2*reel(rand())/RAND_MAX - 1;
  }

  reel x, y, z, vx, vy, vz;
};

struct Particules {

  Particules(int n) : p(n) {}
  void move(reel dT);

  std::vector<Particule> p;

};

#endif