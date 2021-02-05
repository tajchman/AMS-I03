#define IN_MAIN

#include <cstdlib>
#include <iostream>
#include "Calcul_Cuda.hxx"
#include "timer.hxx"

int main(int argc, char **argv)
{  
  T_AllocId = AddTimer("alloc");
  T_CopyId = AddTimer("copy");
  T_InitId = AddTimer("init");
  T_AddId = AddTimer("add");
  T_SommeId = AddTimer("somme");
  T_FreeId = AddTimer("free");
  AddTimer("total");

  int i, n = argc > 1 ? strtol(argv[1], NULL, 10) : 20000000;
  
  Timer & T_total = GetTimer(-1);
  T_total.start();
  
  double v;
  {
    Calcul_Cuda C(n);
    C.init();
    C.addition();
    v = C.somme();
  }

  T_total.stop();

  
  std::cout << "erreur " << v << "\n" 
       << std::endl;

  PrintTimers(std::cout);

  return 0;
}
