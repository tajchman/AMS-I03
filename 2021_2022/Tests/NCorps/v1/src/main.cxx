#include "timer.hxx"
#include "particules.hxx"
#include <cstdio>

int main(const int argc, const char** argv) {

  const int nParticles = (argc > 1 ? atoi(argv[1]) : 16384);
  const int nSteps = 10;
  const reel dt = 0.01;

  Particules p(nParticles);

  Timer t_total;
  t_total.start();

  printf("%10s %10s  \n", "Iteration", "Temps");
  for (int step = 1; step <= nSteps; step++) {

    Timer t;
    t.start();

    p.move(dt);

    t.stop();

    double elapsed = t.elapsed();
    printf("%10d %10.3f s\n", step, t.elapsed());
  }

  t_total.stop();
  double elapsed = t_total.elapsed();
  printf("T total %10.3f s T/iteration moyen %10.3f s\n", elapsed, elapsed/nSteps);
  return 0;
}


