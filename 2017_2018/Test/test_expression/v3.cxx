#include <vector>
#include <iostream>

int main(int argc, char **argv)
{
  double aa, bb;
  bool terme1, terme2;
  int i, n = 20000000, it, nt=10;
  std::vector<double> v(n), a(n), b(n);
  
  std::cout << std::endl << argv[0] << std::endl;
  for (it= 0; it<nt; it++) {
    
    terme1 = true;
    terme2 = true;

    if (terme1) aa = 0.0;
    if (terme2) bb = 0.0;
      
    for (i=0; i<n; i++) {
      v[i] = aa * a[i] + bb * b[i];
    }
  }
  
  return 0;
}
