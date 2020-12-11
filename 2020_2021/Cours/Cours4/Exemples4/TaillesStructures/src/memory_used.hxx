#ifndef __MEMORY_USE
#define __MEMORY_USE

#include <fstream>

class MemoryUsed {

public:
  MemoryUsed();
  ~MemoryUsed();

  void measure(const char *step);
  void setRank(int r) { _rank=r; }

private:

  std::ofstream _f;
  std::string _fNameTemp, _fName;
  long _pid, _rank, _m0;
};

#endif
