#include <iostream>
#include <cstdlib>
#include "timer.hxx"

double Calcul_Pi(std::size_t n);

int main(int argc, char **argv)
{
  Timer T;

  T.start();
  
  std::size_t n = argc > 1 ? strtol(argv[1], NULL, 10) : 10000000L;
  
  double pi = Calcul_Pi(n);

  std::cerr << "Pi (approx) : " << pi << std::endl;
  
  T.stop();
  std::cout << "temps calcul GPU : " << T.elapsed() << " s"
	    << std::endl;

  return 0;
}
