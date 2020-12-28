#include <iostream>

struct Trinome {
  Trinome(double a, double b, double c) : m_a(a), m_b(b), m_c(c) {}
  
  double operator()(double x) { return m_a*x*x + m_b*x + m_c; }

  double m_a, m_b, m_c;
};

  
int main()
{
  double a = 1.0, b = 2.0, c = -1.0;
  
  Trinome T1(a,b,c);
  auto T2 = [a,b,c](double x)->double { return a*x*x + b*x + c; };

  for (double x = 0.0; x < 11.0; x += 1.0)
    std::cerr << x << " " << T1(x) << " " << T2(x) << std::endl;
  
  return 0;
}
