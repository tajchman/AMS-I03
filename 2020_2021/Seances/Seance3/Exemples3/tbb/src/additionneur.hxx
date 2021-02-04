#ifndef _TBB_TASK

#include "tbb/blocked_range.h"
#include <vector>

class Additionneur {

public:
  Additionneur(const std::vector<double> & a, 
               const std::vector<double> & b, 
               std::vector<double> & c)
    :_a(a), _b(b), _c(c)
  {    
  }

  void operator()(const tbb::blocked_range<int>& r) const {
    for (int i = r.begin(); i != r.end(); i++ ) 
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

