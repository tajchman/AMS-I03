#include <cstdlib>
#include <cmath>
#include <iostream>

#include "add.hxx"
#include "timer.hxx"

void addition_CPU(std::vector<double> &w,
		  std::vector<double> &u,
		  std::vector<double> &v)
{
  size_t i, n = u.size();
  for (i=0; i<n; i++)
    w[i] = u[i] + v[i];
}

void init_CPU(std::vector<double> &u,
	      std::vector<double> &v)
{
  size_t i, n = u.size();
  double x;
  for( i = 0; i < n; i++ ) {
    x = double(i);
    u[i] = sin(x)*sin(x);
    v[i] = cos(x)*cos(x);
  }
}

int main(int argc, char **argv)
{
  Timer T_total;
  T_total.start();
  
  size_t i, n = argc > 1 ? strtol(argv[1], NULL, 10) : 20000000;
  size_t bytes = n*sizeof(double);
  double somme;
  
  std::vector<double> u1(n), v1(n), w1(n);
  std::vector<double> u2(n), v2(n), w2(n);
  
  Timer T_CPU, T_GPU;

  T_CPU.start();
  init_CPU(u1, v1);
  addition_CPU(w1, u1, v1);
  T_CPU.stop();

  T_GPU.start();
  init_GPU(u2, v2);
  addition_GPU(w2, u2, v2);
  T_GPU.stop();
  
  somme = 0;
  // double diff_u = 0, diff_v = 0, diff_w = 0;
  for(i=0; i<n; i++) {
    somme += w1[i];
    
    // diff_u = std::max(std::abs(u1[i] - u2[i]), diff_u);
    // diff_v = std::max(std::abs(v1[i] - v2[i]), diff_v);
    // diff_w = std::max(std::abs(w1[i] - w2[i]), diff_w);
  }
  std::cout << "resultat : " << somme/n
    //	    << "\n\terreurs (u) : " << diff_u
    //	    << "\n\terreurs (v) : " << diff_v
    //	    << "\n\terreurs (w) : " << diff_w
	    << std::endl << std::endl;

  std::cout << "temps calcul CPU : " << T_CPU.elapsed() << std::endl;
  std::cout << "temps calcul GPU : " << T_GPU.elapsed() << std::endl;
  
  T_total.stop();
  std::cout << "\ntemps total : " << T_total.elapsed() << std::endl;

  return 0;
}
