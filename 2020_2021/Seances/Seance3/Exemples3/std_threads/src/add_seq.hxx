#ifndef __ADD_SEQ

#include <vector>

class AddFull {

public:
  AddFull(const std::vector<double> & a, 
               const std::vector<double> & b, 
               std::vector<double> & c)
    :_a(a), _b(b), _c(c)
  {    
  }

  void operator()() const {
    int i, n = _c.size();
    for (int i = 0; i != n; i++ ) 
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

