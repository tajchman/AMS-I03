#include "timer.hxx"
#include "particules.hxx"
#include <cstdio>

int main(const int argc, const char** argv) {

  const int nParticles = (argc > 1 ? atoi(argv[1]) : 50000);
  const int nSteps = 10;
  const double dt = 0.01;

  Particules p(nParticles);

  Timer t_total;
  t_total.start();

  printf("\nRemplacement (puissance + 3 divisions) par \n"
         "(racine carrée + 1 division + 5 multiplications)\n\n");
  printf("%10s %10s    %17s \n", "Iteration", "Temps", "Energie cinetique");
  for (int step = 1; step <= nSteps; step++) {

    Timer t;
    t.start();

    p.move(dt);

    t.stop();

    printf("%10d %10.3f s  %16.7g\n", step, t.elapsed(), p.K);
  }

  t_total.stop();
  printf("\nT total    %10.3f s\n\n", t_total.elapsed());
  return 0;
}


