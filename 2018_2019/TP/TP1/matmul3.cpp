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
  int i, j, k, l, kmax, lmax;
  int n = argc > 1 ? strtol(argv[1], nullptr, 10) : 1000;
  int m = argc > 2 ? strtol(argv[2], nullptr, 10) : 2000;
  int p = argc > 3 ? strtol(argv[3], nullptr, 10) : 50;

  Matrice a(n,m), b(m,n);

  const int blocksize = p;
  
  for ( i = 0; i < n; i += blocksize) {
    kmax = i + blocksize; if (kmax > n) kmax = n;
    for (j = 0; j < m; j += blocksize) {
      lmax = j + blocksize; if (lmax > m) lmax = m;
      // transpose the block beginning at [i,j]
      for (k = i; k < kmax; ++k)
        for (l = j; l < lmax; ++l)
          b(l,k) = a(k,l);
    }
  }
        
  return 0;
}
