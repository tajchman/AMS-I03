#include "parameters.h"
#include "GetPot.hxx"

void usage(char *progName) {
  std::cerr << "usage : " << progName << " [list of <option>]\n\n";
  std::cerr << "where <option> may be:\n"
	    << "\t -h|--help : print this message\n"
	    << "\t n=<int>   : number of samples (default : "
	    << 1000L * 200000L << ")\n\n";
  exit(-1);
}

extern "C"
void * parseArgs(int argc, char **argv)
{
  GetPot * P = new GetPot(argc, argv);
  bool help = P->options_contain("h") or P->long_options_contain("help");
  if (help)
    usage(argv[0]);
  return P;
}


extern "C"
double getDouble(void *p, const char * name, double Default)
{
  GetPot * P = (GetPot *) p;
  return (*P)(name, Default);
}

extern "C"
int getInt(void *p, const char * name, int Default)
{
  GetPot * P = (GetPot *) p;
  return (*P)(name, Default);
}

extern "C"
long getLong(void *p, const char * name, long Default)
{
  GetPot * P = (GetPot *) p;
  return (*P)(name, Default);
}

extern "C"
void setDouble(void *p, const char * name, double value)
{
  GetPot * P = (GetPot *) p;
  P->set(name, value, false);
}

extern "C"
void setInt(void *p, const char * name, int value)
{
  GetPot * P = (GetPot *) p;
  P->set(name, value, false);
}

extern "C"
void setLong(void *p, const char * name, long value)
{
  GetPot * P = (GetPot *) p;
  P->set(name, value, false);
}


extern "C"
void freeArgs(void **p)
{
  GetPot * P = (GetPot *) *p;
  delete P;
  *p = NULL;

}

  
  
