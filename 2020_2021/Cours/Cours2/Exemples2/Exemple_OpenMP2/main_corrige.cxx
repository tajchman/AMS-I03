#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

int main() {
  
std::string prefix = " Ici le thread " ;
int iTh ;

# pragma omp parallel private(iTh)
  {
#ifdef _OPENMP
    iTh = omp_get_thread_num () ;
#else
    iTh = 0;
#endif
    std::cout << prefix << iTh << std::endl ;
  }
  return 0;
}
