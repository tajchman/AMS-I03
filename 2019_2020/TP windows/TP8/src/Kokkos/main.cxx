#include <cstring>
#include <iostream>
#include <iomanip>

#include <Kokkos_Core.hpp>

using Device = Kokkos::DefaultExecutionSpace;
typedef Kokkos::View<double **, Device> DeviceArray;
typedef DeviceArray::HostMirror HostArray;

struct Iteration {

  Iteration( DeviceArray & u, DeviceArray & v) 
    : m_u(u), m_v(v) { }

  KOKKOS_INLINE_FUNCTION
  void operator() (int i, int j) const {
    m_v(i,j) = m_u(i,j) + 0.1 * 
	 (-4*m_u(i,j) + m_u(i+1,j) + m_u(i-1,j) + m_u(i, j+1) + m_u(i, j-1)); 
  }

  DeviceArray m_u, m_v;
};

int main(int argc, char** argv) {

  int N = argc > 1 ? strtol(argv[1], NULL, 10) : 10;
  int i,j;

  Kokkos::initialize(argc,argv);

  {
  DeviceArray u("u", N, N);
  HostArray u_host = Kokkos::create_mirror_view(u);
  DeviceArray v("v", N, N);
  HostArray v_host = Kokkos::create_mirror_view(v);

  for (i=0; i<N; i++)
     for (j=0; j<N; j++) {
        u_host(i,j) = i*i + j*j > N*N/2 ? 0 : 1;
        v_host(i,j) = u_host(i,j);
     }
  Kokkos::deep_copy(u, u_host);
 
  for (i=0; i<N; i++) {
     for (j=0; j<N; j++)
        std::cerr << std::setw(8) << u_host(i,j);
     std::cerr << std::endl;
     }
  std::cerr << std::endl;
 
  Kokkos::Timer timer;

  Kokkos::MDRangePolicy< Kokkos::Rank<2> > Rg({1,1}, {N-1,N-1});
  Iteration It(u, v);
  Kokkos::parallel_for(Rg, It);

  Kokkos::deep_copy(v_host, v);
  for (i=0; i<N; i++) {
     for (j=0; j<N; j++)
        std::cerr << std::setw(8) << v_host(i,j);
     std::cerr << std::endl;
     }
  std::cerr << std::endl;
 
  }

  Kokkos::finalize();

  return 0;
}
