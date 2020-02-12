#include <iostream>
#include <vector>

struct X {
  int z[2];
};
  
int main()
{
  std::vector<X> v(10);
  int a = 1;

  std::vector<X>::iterator p;
  for (p = v.begin(); p != v.end(); p++) {
    p->z[0] = a;
    p->z[1] = p->z[0] * 3 + 1;
    a = a * 2;
  }

  std::vector<X>::const_iterator q;
  for (q = v.begin(); q != v.end(); q++) {
    std::cerr << q->z[1] << std::endl;
  }

  return 0;
}

