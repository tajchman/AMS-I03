#include <vector>
#include <cstring>
#include <cmath>

class Matrice {

public:
  Matrice (int n, int m) : m_n(n), m_m(m), m_coefs(n*m) {
    int i; for (i=0; i<n*m; i++) m_coefs[i] = 0.0;
  }
  double operator()(int i,int j) const { return m_coefs[m_m*i + j]; }
  double & operator()(int i,int j) { return m_coefs[m_m*i + j]; }
private:
  int m_n, m_m;
  std::vector<double> m_coefs;
};

int main(int argc, char **argv)
{
  int i, j;
  int n = argc > 1 ? strtol(argv[1], nullptr, 10) : 1000;
  int m = argc > 2 ? strtol(argv[2], nullptr, 10) : 2000;

  Matrice a(n,m), b(m,n);

  for (j=0; j<m; j++)
    for (i=0; i<n; i++)
      b(j,i) = a(i,j);

  return 0;
}
