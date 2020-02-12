#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
#include "timer.hxx"

void maxlocal(std::vector<double> & v,
	      std::vector<int> & imax, std::size_t &nmax) {

  std::size_t i, n = v.size();
  std::size_t smax = imax.size();
  nmax = 0;

  //  #pragma omp parallel for
  for (i=1; i<n-1; i++) {
    if ((v[i-1] < v[i]) && (v[i] > v[i+1])) {
      //      #pragma omp critical
      {
      imax[nmax] = i;
      nmax += 1;
      if (nmax >= smax) break;
      }
    }
  }
  
}

int main() {

  Timer T0;
  T0.start();
  
  const std::size_t N = 100000000;
  const std::size_t M = 100000;
  std::size_t i, nmax;
  std::vector<double> v(N);
  std::vector<int> idx(M, -1);
  double x;
  
  T0.stop();
  std::cerr << "T alloc  : " << T0.elapsed() << std::endl;
  
  Timer T1;
  T1.start();
  
  #pragma omp parallel for
  for (i=0; i<N; i++)
    v[i] = sin((3.1415926 * i * 100000)/N);
  T1.stop();
  std::cerr << "T init   : " << T1.elapsed() << std::endl;

  
  Timer T2;
  T2.start();

  std::cerr << "\tdebut recherche max locaux" << std::endl;
  
  nmax = 0;
  maxlocal(v, idx, nmax);
  
  std::cerr << "\tfin recherche max locaux nmax = " << nmax << std::endl;
  
  T2.stop();
  std::cerr << "T calcul : " << T2.elapsed() << std::endl;

  Timer T3;
  T3.start();
  
  std::ofstream f("max_fine_grain.txt");
  for (auto & k : idx) {
    if (k < 0) break;
    f << k << "  " << v[k] << std::endl;
  }
  
  T3.stop();
  std::cerr << "T sauve  : " << T3.elapsed() << std::endl;

  std::cerr << "T total  : "
	    << T0.elapsed() + T1.elapsed() + T2.elapsed() + T3.elapsed()
	    << std::endl;
  
}
