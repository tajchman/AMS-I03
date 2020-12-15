#include "memory_used.hxx"
#include <cstring>
#include <sstream>

long find_info(const char *name)
{
  long to_read = 8192;
  char buffer[to_read];
  int read;

  long value;
  short found = 0;
  char* search_result;

  char delims[] = "\n";

  FILE * procfile = fopen("/proc/self/status", "r");
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
  _pid = find_info("Pid:");
}

MemoryUsed::~MemoryUsed()
{
  if (_f.is_open())
    _f.close();
}

void MemoryUsed::initMeasure()
{
  _m0 = find_info("VmRSS:");
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

  long vmrss_kb = find_info("VmRSS:");
  _f << step << ": " << vmrss_kb - _m0 << "kb" << std::endl;
}

