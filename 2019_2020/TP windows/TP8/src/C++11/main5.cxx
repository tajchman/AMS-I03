#include <iostream>

struct myFunction {
  double operator()(double x) { return x*x; }
};

  
int main()
{
  myFunction sqr_1;
  std::cerr << sqr_1(2.5) << std::endl;
  
  auto sqr_2 = [](double x) { return x*x; };
  std::cerr << sqr_2(2.5) << std::endl;

  return 0;
}
