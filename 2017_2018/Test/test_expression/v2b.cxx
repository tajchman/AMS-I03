#include <vector>
#include <iostream>

int main(int argc, char **argv)
{
  double aa, bb;
  int i, n = 20000000, it, nt=10;
  std::vector<double> v(n), a(n), b(n);
  
  std::cout << std::endl << argv[0] << std::endl;
  for (it= 0; it<nt; it++) {
         
     for (i=0; i<n; i++) {
       v[i] = aa * a[i];
     }
    
     for (i=0; i<n; i++) {
       v[i] += bb * b[i];
     }
  }
  
  return 0;
}
