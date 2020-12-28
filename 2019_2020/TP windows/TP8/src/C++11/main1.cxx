#include <iostream>
#include <vector>

struct X {
  int z[2];
};
  
int main()
{
  std::vector<X> v(10);
  int a = 1;
  
  int i;
  for (i=0; i<v.size(); i++) {
    v[i].z[0] = a;
    v[i].z[1] = v[i].z[0] * 3 + 1;
    a = a * 2;
  }

  for (i=0; i<v.size(); i++) {
    std::cerr << v[i].z[1] << std::endl;
  }

  return 0;
}

