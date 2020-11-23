#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

int main() {
  
std::string prefix = " Ici le thread " ;

# pragma omp parallel
  {
    int iTh ;
# ifdef _OPENMP
    iTh = omp_get_thread_num () ;
# else
    iTh = 0;
# endif
    std::cerr << prefix << iTh << std::endl ;
  }
  return 0;
}
