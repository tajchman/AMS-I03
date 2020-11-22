#include "timer_papi.hxx"
#include <stdexcept>

bool Timer::active = false;
int Timer::EventSet = PAPI_NULL;

void TimerInit()
{
  int retval = PAPI_library_init(PAPI_VER_CURRENT);
  
  if (retval != PAPI_VER_CURRENT && retval > 0)
    throw std::runtime_error("PAPI library version mismatch!");

  if (retval < 0)
    throw std::runtime_error("PAPI initialisation error");

  std::cout << "PAPI Version Number: "
            << PAPI_VERSION_MAJOR(retval) << '.' 
            << PAPI_VERSION_MINOR(retval) << '.'
            << PAPI_VERSION_REVISION(retval) << std::endl;

  Timer::EventSet = PAPI_NULL;
  if (PAPI_create_eventset(&Timer::EventSet) != PAPI_OK)
    throw std::runtime_error("PAPI create event set error");

  if (PAPI_add_event(Timer::EventSet, PAPI_TOT_CYC) != PAPI_OK)
    throw std::runtime_error("PAPI add event set error");

  Timer::active = true;
}
