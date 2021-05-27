#include "timer.hxx"
#include "particules.hxx"
#include <papi.h> 

int main(const int argc, const char** argv) {

  const int nParticles = (argc > 1 ? atoi(argv[1]) : 16384);
  const int nSteps = 10;  // Duration of test
  const reel dt = 0.01; // Particle propagation time step

  Particules p(nParticles);

  double rate = 0, dRate = 0; // Benchmarking data
  const int skipSteps = 3; // Skip first iteration is warm-up on Xeon Phi coprocessor

  Timer t;

  for (int step = 1; step <= nSteps; step++) {

    t.start();

    p.move(dt);

    t.stop();

    double elapsed = t.elapsed();
  }
}


