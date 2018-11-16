#include <cstring>
#include <cmath>
#include "Matrice.hpp"
#include "timer.hpp"

struct F {
    inline double operator()(int i, int j)
{
  return 1.0/(i+j*2+1);
}
};

int main(int argc, char **argv)
{
  Timer t_total;
  t_total.start();

  int i, j, k, l, kmax, lmax;
  int n = argc > 1 ? strtol(argv[1], nullptr, 10) : 10000;
  int m = argc > 2 ? strtol(argv[2], nullptr, 10) : 10000;
  int p = argc > 3 ? strtol(argv[3], nullptr, 10) : 50;

  std::cerr << "transpose 3 : taille matrice = (" << n << ", " << m << ") bloc = " << p << std::endl;

{
  Timer t;
  t.start();

  F f;
  Matrice a(n,m), b(m,n);
  init(a, f);

  t.stop();
  std::cerr << "init    time : " << t.elapsed() << " s"  << std::endl;

  t.reinit();
  t.start();

  const int blocksize = p;
  
  for ( i = 0; i < n; i += blocksize) {
    kmax = i + blocksize; if (kmax > n) kmax = n;
    for (j = 0; j < m; j += blocksize) {
      lmax = j + blocksize; if (lmax > m) lmax = m;

      for (k = i; k < kmax; ++k)
        for (l = j; l < lmax; ++l)
          b(l,k) = a(k,l);
    }
  }

  t.stop();

  if (n<10 && m<10) {
    std::cout << "A" << std::endl << a << std::endl;
    std::cout << "B" << std::endl << b << std::endl;
  }

  std::cerr << "compute time : " << t.elapsed() << " s"  << std::endl;
  }

  t_total.stop();
  std::cerr << "total time   : " << t_total.elapsed() << " s"  << std::endl;

  return 0;
}