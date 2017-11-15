#include <stdio.h>
#include "util.hpp"

void wait() {
  puts("Press return to continue ... ");
  getchar();
}

#ifdef __linux__
#include <unistd.h>

size_t memavail(double factor)
{
  if (factor > 1.0 || factor < 0.0) return 0L;
  
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  return (size_t) (pages * page_size * factor);
}

#elif _WIN32

#include <windows.h>

size_t memavail(double factor)
{
  if (factor > 1.0 || factor < 0.0) return 0L;
  
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return (size_t) (status.ullTotalPhys * factor);
}

#else

#endif
