#include "memory_used.hxx"
#include <cstring>
#include <sstream>
#include <iostream>
#include <string>
#include <cmath>

long find_info(const char *fileName, const char *name)
{
  long to_read = 8192;
  char buffer[to_read];
  int read;

  long value;
  short found = 0;
  char* search_result;

  char delims[] = "\n";

  FILE * procfile = fopen(fileName, "r");
  read = fread(buffer, sizeof(char), to_read, procfile);
  fclose(procfile);
  char* line = strtok(buffer, delims);

  while (line != NULL && found == 0)
    {
      search_result = strstr(line, name);
      if (search_result != NULL)
      {
 	      sscanf(line, "%*s %ld", &value);
	      found = 1;
      }

      line = strtok(NULL, delims);
    }
  return value;
}

MemoryUsed::MemoryUsed() : _rank(0)
{
  _pid = find_info("/proc/self/status", "Pid:");
}

MemoryUsed::~MemoryUsed()
{
  if (_f.is_open())
    _f.close();
}

void MemoryUsed::initMeasure()
{
  _m0 = find_info("/proc/self/status", "VmRSS:");
}

void MemoryUsed::endMeasure(const char *step)
{

  if (!_f.is_open())
  {
    std::ostringstream s;
    s << "memory_" << _rank << "_" << _size << ".txt";
    _fName = s.str();
    _f.open(_fName.c_str());
  }

  long vmrss_kb = find_info("/proc/self/status", "VmRSS:");
  _f << step << ": " << vmrss_kb - _m0 << " kb" << std::endl;
}

void GetMeasure(const char *step, int size, int & mean, int & stddev)
{
  long to_read = 8192;
  char buffer[to_read];
  int read;

  long value;
  short found = 0;
  char* search_result;

  char delims[] = "\n";


  int i;
  char s[100];
  float moy = 0.0, moy2 = 0.0;
  for (i=0; i<size; i++) {

    float val;
    sprintf(s, "memory_%d_%d.txt", i, size);
    val = find_info(s, step);
    moy += val;
    moy2 += val*val;
  }
  
  moy /= size;
  moy2 = std::sqrt(moy2/size - moy*moy);

  mean = int(moy);
  stddev = int(moy2);
}