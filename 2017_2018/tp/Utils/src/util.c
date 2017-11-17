#include <stdio.h>
#include "util.h"

void waitKey() {
  puts("\nPress return to continue ... ");
  getchar();
}

<<<<<<< HEAD:2017_2018/tp/Ex2/src/util.cpp
#if defined(__linux__) || defined(__APPLE__)
=======

#if  defined(__linux__) || defined(__APPLE__)
>>>>>>> faeba1d4d9019c3e355863bd16d42d69abb5fc70:2017_2018/tp/Utils/src/util.c

#include <unistd.h>

size_t memavail(double factor)
{
  if (factor > 1.0 || factor < 0.0) return 0L;
  
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  return (size_t) (pages * page_size * factor);
}

<<<<<<< HEAD:2017_2018/tp/Ex2/src/util.cpp
#elif defined(_WIN32) || defined(__CYGWIN__) 
=======
#elif defined(_WIN32) || defined(__CYGWIN__)
>>>>>>> faeba1d4d9019c3e355863bd16d42d69abb5fc70:2017_2018/tp/Utils/src/util.c

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
