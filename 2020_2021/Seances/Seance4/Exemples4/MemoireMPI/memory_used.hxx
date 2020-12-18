#ifndef __MEMORY_USE
#define __MEMORY_USE

#include <fstream>

class MemoryUsed {

public:
  MemoryUsed();
  ~MemoryUsed();

  void initMeasure();
  void endMeasure(const char *step);
  void setRank(int r, int s) { _rank=r, _size=s; }

private:

  std::ofstream _f;
  std::string _fNameTemp, _fName;
  long _pid, _rank, _size, _m0;
};

void GetMeasure(const char *step, int size, int & mean, int & stddev);

#endif
