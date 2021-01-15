#include <iostream>
#include <iomanip>
#include "matrice.hxx"
#include "timer.hxx"

int main(int argc, char **argv)
{
  int kmax = 10;
  int n = argc > 1 ? strtol(argv[1], NULL, 10) : 10;

  AddTimer("allocation");
  AddTimer("copie memoire");
  AddTimer("identite");
  AddTimer("norme2");
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
  
  std::cout << "Iteration" << std::setw(8) << " "
            << "variation" << std::endl;

  for (int k=0; k<kmax; k++) {
    multiply(Temp, Ap, Am);
    Ap = Temp;
  
    B += Ap;
  
    double delta = Ap.norm2();
    std::cout<< std::setw(9) << k 
             << "    " << std::setw(13) << std::scientific << delta
             << '\r';
    
    if (delta < 1e-5)
      break;
  }
  std::cout << std::endl << std::endl;

  Temp.name() = "Verification";
  multiply(Temp, A, B);
  if (n <= 10) {
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    std::cout << Temp << std::endl;
  }

  Temp -= Id;

  std::cout << "\nErreur sur l'inverse = " << Temp.norm2() << std::endl;

  T_total.stop();
  PrintTimers(std::cout);

  return 0;

}
