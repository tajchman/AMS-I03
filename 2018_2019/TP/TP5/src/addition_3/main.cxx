#include <cstdlib>
#include <iostream>
#include "calcul.hxx"
#include "timer.hxx"

int main(int argc, char **argv)
{
  Timer T_total;
  T_total.start();
  
  size_t i, n = argc > 1 ? strtol(argv[1], NULL, 10) : 20000000;
  
  Timer T_CPU, T_GPU;

  T_CPU.start();
  
  Calcul_CPU C1(n);
  C1.addition();
  double v1 = C1.verification();
  
  T_CPU.stop();
  std::cout << "temps calcul CPU : " << T_CPU.elapsed() << std::endl << std::endl;

  T_GPU.start();
  
  Calcul_GPU C2(n);
  C2.addition();
  double v2 = C2.verification();
  
  T_GPU.stop();
  std::cout << "temps calcul GPU : " << T_GPU.elapsed() << std::endl;
  
  std::cerr << "\nresultat : \n"
	    << "\t CPU " << v1 << "\n"
	    << "\t GPU " << v2 << "\n"
    //	    << "\n\terreurs (u) : " << diff_u
    //	    << "\n\terreurs (v) : " << diff_v
    //	    << "\n\terreurs (w) : " << diff_w
	    << std::endl;
  
  T_total.stop();
  std::cout << "\ntemps total : " << T_total.elapsed() << std::endl;

  return 0;
}
