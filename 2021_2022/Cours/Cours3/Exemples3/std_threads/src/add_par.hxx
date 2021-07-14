#ifndef _ADD_TBB

#include <vector>
#include <iostream>

class AddPartial {

public:
  AddPartial(const std::vector<double> & a, 
             const std::vector<double> & b, 
             std::vector<double> & c)
    :_a(a), _b(b), _c(c)
  {    
  }

  void operator()(int iStart, int iEnd) const {
    for (int i = iStart; i != iEnd; i++ ) 
    {
      _c[i] = _a[i] + _b[i];
    }
  }

private:
  const std::vector<double> & _a; 
  const std::vector<double> & _b;
  std::vector<double> & _c;
};

#endif

