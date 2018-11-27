#include "pause.hpp"
#include <thread>
#include <chrono>

void pause(int d) {
//  std::this_thread::sleep_for(std::chrono::microseconds(n));
  volatile double v = 0;
  for(int n=0; n<d; ++n)
     for(int m=0; m<d; ++m)
         v += v*n*m;

}

