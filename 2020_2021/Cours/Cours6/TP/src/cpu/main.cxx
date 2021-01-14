#include <iostream>
#include "matrice.hxx"
#include "timer.hxx"

int main(int argc, char **argv)
{
  int kmax = 200;
  int n = argc > 1 ? strtol(argv[1], NULL, 10) : 10;

  AddTimer("allocation");
  AddTimer("copie memoire");
  AddTimer("identite");
  AddTimer("random");
  AddTimer("addition");
  AddTimer("multiplication");
  AddTimer("total");

  Timer & T_total = GetTimer(-1);
  T_total.start();

  Matrice Id(n, n, "Id");
  Matrice A(n, n, "A");
  Matrice B(n, n, "inverse(A)");
  Matrice Am(n, n, "Id-A");
  Matrice Ap(n, n, "(Id-A)^p");
  Matrice Temp(n, n,"Temp");

  Id.Identite();
  A.Random(1.0/n);
 
  Am = Id;
  Am -= A;
 
  B = Id;
  Ap = Id;
  
  for (int k=0; k<kmax; k++) {
    multiply(Temp, Ap, Am);
    Ap = Temp;
  
    B += Ap;
  
    double delta = Ap.norm2();
    std::cout << k << " " << delta << std::endl;
    if (delta < 1e-7)
      break;
  }

  Temp.name() = "Verification";
  multiply(Temp, A, B);
  if (n <= 10) {
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << Temp << std::endl;
  }

  Temp -= Id;

  std::cout << "\nErreur sur l'inverse " << Temp.norm2() << std::endl;

  T_total.stop();
  PrintTimers(std::cout);

  return 0;

}
