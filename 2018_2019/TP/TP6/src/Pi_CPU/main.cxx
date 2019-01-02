#include <iostream>
#include "timer.hxx"

double Calcul_Pi(std::size_t n);

int main(int argc, char **argv)
{
  Timer T_CPU;

  T_CPU.start();
  
  std::size_t n = argc > 1 ? strtol(argv[1], NULL, 10) : 100000000;
  
  double pi = Calcul_Pi(n);

  std::cerr << "Pi (approx) : " << pi << std::endl;
  
  T_CPU.stop();
  std::cout << "temps calcul CPU : " << T_CPU.elapsed() << " s"
	    << std::endl;

  return 0;
}
