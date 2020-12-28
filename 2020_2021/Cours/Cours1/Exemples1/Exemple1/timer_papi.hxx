#include <cstdlib>
#include <ctime>
#include <string>
#include <stdexcept>
#include <papi.h>
#include <iostream>

void TimerInit();

class Timer
{
  public:
    static bool active;
    static int EventSet;

    Timer(const char *name = NULL) {
        if (name)
          _name = name;
        if (!active)
            TimerInit();
        _elapsed = 0.0;
    }

    void reset()
    {
      _elapsed = 0.0;
    }

    void start()
    {
      if (PAPI_start(EventSet) != PAPI_OK)
        throw std::runtime_error(_name + " PAPI start timer error");
    }

    void stop()
    {
      long_long d;
      if (PAPI_stop(EventSet, &d) != PAPI_OK)
         throw std::runtime_error(_name + " PAPI stop timer error");
      _elapsed += d;
    }

    long_long elapsed()
    {
       return _elapsed;
    }
  private:
     std::string _name;
     long_long _start_usec, _end_usec;
     long _elapsed;
};

 