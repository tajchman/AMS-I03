#include <vector>

class Vecteur {
public:
  Vecteur (int n) : m_coeffs(n) {
  }

  double & operator()(int i) { return m_coeffs[i]; }
  double operator()(int i) const { return m_coeffs[i]; }
  void operator=(double d) {
    int i, n = size();
    for (i=0; i<n; i++)
      (*this)(i) = d;
  }
  
  int size() const { return int(m_coeffs.size()); }
  double normalise() ;
  
private:

  std::vector<double> m_coeffs;
};
