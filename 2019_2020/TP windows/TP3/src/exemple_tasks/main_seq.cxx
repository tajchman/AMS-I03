#include <list>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cmath>

#include "sin.hxx"
#include "timer.hxx"

int main()
{
  
  std::srand(0);
  
  std::list<std::pair<double, double>> L;
  double x = 0.001;
  
  for (x = 0.01; x < 10.0; x *= (1.0 + 0.2*double(std::rand())/RAND_MAX)) {
    L.push_back(std::pair<double, double>{x, 0.0});
  }
  std::cerr << "Liste de " << L.size() << " elements" << std::endl;
  
  set_terms(40);
 
  Timer T;
  T.start();

  for (auto e = L.begin(); e != L.end(); e++)
      e->second = sinus_taylor(e->first);
  
  T.stop();

  std::cerr  << "temps de calcul : " << T.elapsed() << std::endl;

  double erreur = 0.0;
  for (const auto & e:L)
    erreur += e.second - sin(e.first);

  std::cerr << "erreur = " << std::setw(12) <<  erreur << std::endl;
  return 0;
}
