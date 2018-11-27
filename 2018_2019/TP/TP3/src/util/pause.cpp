#include "pause.hpp"
#include <thread>
#include <chrono>

void pause(int n) {
  std::this_thread::sleep_for(std::chrono::milliseconds{n});
}
