#include <iostream>
#include <vector>

struct X {
  int z[2];
};
  
int main()
{
  std::vector<X> v(10);
  int a = 1;

  for (auto & c : v) {
    c.z[0] = a;
    c.z[1] = c.z[0] * 3 + 1;
    a = a * 2;
  }

  for (const auto & c : v) {
    std::cerr << c.z[1] << std::endl;
  }

}
