#include <iostream>
#include <vector>
#include <cstdlib>
#include <time.h>
#include "timer.hxx"
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char **argv) {

  Timer T_total, T_init, T_calcul;
  T_total.start();

  double moy, var;

  {
  T_init.start();
  
  int i, n = argc > 1 ? strtol(argv[1], NULL, 10) : 100000000;
  std::vector<double> u(n);

  srand (time(NULL));
  srand (0);
  for (i=0; i<n; i++)
    u[i] = 0.5 + double(rand())/RAND_MAX;

  T_init.stop();
  T_calcul.start();

  double s = 0 , s_partiel ;
  double s2 = 0 , s2_partiel ;

  # pragma omp parallel default(shared) private ( s_partiel , s2_partiel )
  {
    s_partiel = 0.0; s2_partiel = 0.0;
  
    # pragma omp for
    for (i =0; i<n ; i++) {
      s_partiel += u[i];
      s2_partiel += u[i]*u[i];
    }

    # pragma omp critical
    {
      s += s_partiel; 
      s2 += s2_partiel;
    }
/*
    #pragma omp atomic
      s += s_partiel; 
    #pragma omp atomic
      s2 += s2_partiel; 
*/
  }

  moy = s / n ; 
  var = s2 /n - moy * moy ;

  T_calcul.stop();

  }
  T_total.stop();

  std::cout << "moyenne : " << moy << " variance " << var << std::endl;
  std::cout << "temps init   CPU : " << T_init.elapsed() << " s" << std::endl;
  std::cout << "temps calcul CPU : " << T_calcul.elapsed() << " s" << std::endl;
  std::cout << std::endl;
  std::cout << "temps total  CPU : " << T_total.elapsed() << " s" << std::endl;
  return 0;
}
