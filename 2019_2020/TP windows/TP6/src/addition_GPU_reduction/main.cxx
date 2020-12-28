#include <cstdlib>
#include <iostream>
#include "calcul.hxx"
#include "timer.hxx"

int main(int argc, char **argv)
{
  size_t i, n = argc > 1 ? strtol(argv[1], NULL, 10) : 20000000;
  
  Timer T_GPU;

  T_GPU.start();
  
  Calcul_GPU C(n);
  C.addition();
  double v = C.verification();
  
  T_GPU.stop();
  std::cout << "temps calcul GPU : " << T_GPU.elapsed() << std::endl;
  
  std::cerr << "erreurs GPU " << v << "\n"
    //	    << "\n\terreurs (u) : " << diff_u
    //	    << "\n\terreurs (v) : " << diff_v
    //	    << "\n\terreurs (w) : " << diff_w
	    << std::endl;
  
  return 0;
}
