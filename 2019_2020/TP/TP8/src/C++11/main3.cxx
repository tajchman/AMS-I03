#include <iostream>
#include <vector>

struct X {
  int z[2];
};
  
int main()
{
  std::vector<X> v(10);
  int a = 1;

  for (auto p = v.begin(); p != v.end(); p++) {
    p->z[0] = a;
    p->z[1] = p->z[0] * 3 + 1;
    a = a * 2;
  }

  for (auto p = v.begin(); p != v.end(); p++) {
    std::cerr << p->z[1] << std::endl;
  }

}
