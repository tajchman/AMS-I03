#include <cstring>
#include<Kokkos_Core.hpp>

int main(int argc, char* argv[]) {

   int N = argc > 1 ? : strtol(argv[1], 10, NULL) : 100;
   int it, n_it = 10;

   Kokkos::initialize(argc,argv);

   Kokkos::MDRangePolicy< Kokkos::Rank<2> > Rg({0,0}, {N,N})

   Kokkos::View<double**> u("u", N, N), v("v", N, N);
	
   Kokkos::Timer timer;

  for (it = 0; it < n_it; it++ ) {

    Kokkos::parallel_for( Rg, KOKKOS_LAMBDA (int i, int j) {
       v(i,j) = u(i,j) - 0.25 * 
              (u(i+1,j) + u(i-1,j) + u(i, j+1) + u(i, j-1)); 
    }

  }
  Kokkos::finalize();

  return 0;
}
