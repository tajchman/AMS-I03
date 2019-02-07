#include<Kokkos_Core.hpp>
#include<cstdio> 

struct It1 {

  void operator()(const int &i) {
     std::cerr << "Greeting from iteration " << i << std::endl;
  }

};

int main(int argc, char* argv[]) {

   Kokkos::initialize(argc,argv);

   int N = argc > 1 ? atoi(argv[1]) : 10;

   It1 I;
   Kokkos::parallel_for("Loop1", N, I);

   Kokkos::finalize();

   return 0;
}
