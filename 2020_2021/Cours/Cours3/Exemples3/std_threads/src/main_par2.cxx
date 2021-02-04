
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

class Worker {

public:
  Worker() {}
  virtual ~Worker() { stop_thread(); }
  inline void stop_thread() {
    exiting.store(true);
    has_work.store(true);
    if (thread.joinable()) {
      thread.join();
    }
  }

  virtual void apply() = 0;

  inline void work() {
    has_work.store(true);
  }

  inline void finish() {
    while (has_work.load()) {
    }
  }

private:
  std::atomic<bool> has_work{false};

  std::atomic<bool> exiting{false};
  std::atomic<bool> thread_started{false};

  std::thread thread = std::thread([this] {
    thread_started.store(true);
    while (true) {
      while (!has_work.load()) {
        if (exiting.load()) {
          return;
        }
      }
      apply();
      has_work.store(false);
    }
  });
};

class AddPartial : public Worker
{
public:
  AddPartial() : _a(nullptr), _b(nullptr), _c(nullptr) {}

  AddPartial(std::vector<double> & a, 
             std::vector<double> & b,
             std::vector<double> & c,
             int i0, int i1) : _a(&a), _b(&b), _c(&c), _i0(i0), _i1(i1) {}

  
  void apply()
  {
    int i;
    std::vector<double> & a = *_a;
    std::vector<double> & b = *_b;
    std::vector<double> & c = *_c;
    for (i=_i0; i<_i1; i++)
      c[i] = a[i] + b[i];
  }

private:
  AddPartial(const AddPartial &other);
  std::vector<double> * _a; 
  std::vector<double> * _b;
  std::vector<double> * _c;
  int _i0, _i1;
};


int main() {

  size_t n = 1000000000;
  int iTh, nThreads = 6;

  std::vector<double> a(n, 1); 
  std::vector<double> b(n, 2);

  std::vector<AddPartial *> Adds;

  int i0, i1 = 0;
  size_t dn = n / nThreads;
  for (iTh = 0; iTh < nThreads-1; iTh++) {
    i0 = i1;
    i1 = i0 + dn;
    Adds.push_back(new AddPartial(a, b, a, i0, i1));
  }

  i0 = i1;
  i1 = n;
  Adds.push_back(new AddPartial(a, b, a, i0, i1));

  for (auto & Add : Adds) {
    Add->work();
  }

  for (auto & Add : Adds) {
    Add->finish();
  }
  
  for (auto & x : a)
    std::cout << x << std::endl;

  return 0;
}

