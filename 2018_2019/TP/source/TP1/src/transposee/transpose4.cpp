#include <cstring>
#include <cmath>
#include "Matrice.hpp"
#include "MatriceBloc.hpp"
#include "timer.hpp"

void init(MatriceBloc &A)
{
  int i,j,k,l, nb = A.n(), mb = A.m(), p = A.p(), q = A.q();
  double s;
  
  for (i=0; i<nb; i++) {
    for (j=0; j<mb; j++) {
      
      Matrice & bA = A(i,j);
      
      for (k=0; k<p; k++)
        for (l=0; l<q; l++) {
          bA(k,l) = 1.0/(i*p+k+(j*q+l)*2+1);
        }
    }
    
  }
  
}


int main(int argc, char **argv)
{
  Timer t_total;
  t_total.start();

  int n = argc > 1 ? strtol(argv[1], nullptr, 10) : 10000;
  int m = argc > 2 ? strtol(argv[2], nullptr, 10) : n;
  int p = argc > 3 ? strtol(argv[3], nullptr, 10) : 50;
  int q = argc > 4 ? strtol(argv[4], nullptr, 10) : p;
  
  std::cerr << "transpose 4 : taille matrice = (" << n << ", " << m
            << ") bloc = " << "(" << p << "x" << q << ")" << std::endl;

  Timer t;
  {
    t.start();
  
    MatriceBloc A(n/p,m/q, p, q), B(m/p,n/q, p, q);
    
    init(A);
  
    t.stop();
    std::cerr << "init    time : " << t.elapsed() << " s" << std::endl;
  
    t.reinit();
    t.start();

    int i,j,k,l, nb = A.n(), mb = A.m(), p = A.p(), q = A.q();
    for (i=0; i<nb; i++) {
      for (j=0; j<mb; j++) {
      
        const Matrice & bA = A(i,j);
        Matrice & bB = B(j,i);
      
        for (k=0; k<p; k++)
          for (l=0; l<q; l++)
            bB(l,k) = bA(k,l);
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
