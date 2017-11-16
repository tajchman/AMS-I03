#include <cstddef>

class Vecteur {
public:
  Vecteur(size_t n);
  ~Vecteur();
  double & operator[](size_t i)  { return m_v[i]; }
  double operator[](size_t i) const { return m_v[i]; }
  
protected:
  size_t m_n;
  double * m_v;
};
