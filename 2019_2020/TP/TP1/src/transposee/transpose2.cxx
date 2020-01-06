#include <cstring>
#include <cstdlib>
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

  int n = argc > 1 ? strtol(argv[1], NULL, 10) : 10000;
  int m = argc > 2 ? strtol(argv[2], NULL, 10) : n;

  std::cerr << "transpose 2 : taille matrice = (" << n << ", " << m << ")" << std::endl;

  Timer t;
  {
  t.start();

    Matrice A(n,m), B(m,n);

    init(A);

  t.stop();
  std::cerr << "init    time : " << t.elapsed()  << " s" << std::endl;

  t.reinit();
  t.start();

  int i, j;
  for (j=0; j<m; j++)
    for (i=0; i<n; i++)
      B(j,i) = A(i,j);

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
