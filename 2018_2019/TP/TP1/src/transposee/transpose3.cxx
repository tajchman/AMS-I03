#include <cstring>
#include <cmath>
#include "Matrice.hxx"
#include "timer.hxx"

void init(Matrice &A)
{
  int i,j, n = A.n(), m = A.m();
  for (i=0; i<n; i++)
    for (j=0; j<m; j++)
      A(i,j) = 1.0/(i+j*2+1);
}

int main(int argc, char **argv)
{
  Timer t_total;
  t_total.start();

  int n = argc > 1 ? strtol(argv[1], nullptr, 10) : 10000;
  int m = argc > 2 ? strtol(argv[2], nullptr, 10) : n;
  int p = argc > 3 ? strtol(argv[3], nullptr, 10) : 50;
  int q = argc > 4 ? strtol(argv[4], nullptr, 10) : p;

  std::cerr << "transpose 3 : taille matrice = (" << n << ", " << m 
            << ") bloc = " << "(" << p << "x" << q << ")" << std::endl;

  Timer t;
{
  t.start();

  Matrice A(n,m), B(m,n);

    init(A);

  t.stop();
  std::cerr << "init    time : " << t.elapsed() << " s"  << std::endl;

  t.reinit();
  t.start();

  int i, j, k, l, kmax, lmax;
  for ( i = 0; i < n; i += p) {
    kmax = i + p; if (kmax > n) kmax = n;
    for (j = 0; j < m; j += p) {
      lmax = j + p; if (lmax > m) lmax = m;

      for (k = i; k < kmax; ++k)
        for (l = j; l < lmax; ++l)
          B(l,k) = A(k,l);
    }
  }

  t.stop();

  if (n<10 && m<10) {
      std::cout << "A" << std::endl << A << std::endl;
      std::cout << "B" << std::endl << B << std::endl;
  }

  std::cerr << "compute time : " << t.elapsed() << " s"  << std::endl;
  }

  t_total.stop();
  std::cerr << "total time   : " << t_total.elapsed() << " s"  << std::endl;

  return 0;
}
