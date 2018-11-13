#include <vector>

class Vecteur {
public:
  Vecteur (int n) : m_coeffs(n) {
  }

  double & operator()(int i) { return m_coeffs[i]; }
  double operator()(int i) const { return m_coeffs[i]; }

  int size() { return m_coeffs.size(); }
  double normalise() ;
  
private:

  std::vector<double> m_coeffs;
};
