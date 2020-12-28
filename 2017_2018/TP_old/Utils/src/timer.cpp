#include <chrono>
#include <iostream>

using namespace std::chrono;

extern "C"
double time_precision()
{
  return (double) std::chrono::high_resolution_clock::period::num
    / std::chrono::high_resolution_clock::period::den;
}

extern "C"
void * start()
{
 auto * t = new time_point<high_resolution_clock>;
  *t = high_resolution_clock::now();
  return t;
}

extern "C"
double elapsed(void * v) {
  time_point<high_resolution_clock> * start =
    static_cast<time_point<high_resolution_clock>*>(v);
  auto end = high_resolution_clock::now();
  duration<double> diff = end- *start;
  return diff.count();
}

extern "C"
void stop(void *v) {
  time_point<high_resolution_clock> * start =
    static_cast<time_point<high_resolution_clock>*>(v);
  delete start;
}
