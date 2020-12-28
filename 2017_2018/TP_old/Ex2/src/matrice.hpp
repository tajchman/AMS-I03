#include <cstddef>

class Matrice {
public:
  Matrice(size_t n, size_t m);
  ~Matrice();
  double * operator[](size_t i)  { return m_v + i*m_m; }
  const double * operator[](size_t i) const { return m_v + i*m_m; }

protected:
  size_t m_n, m_m;
  double * m_v;
};
