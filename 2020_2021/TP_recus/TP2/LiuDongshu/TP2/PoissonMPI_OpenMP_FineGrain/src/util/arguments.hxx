#ifndef _ARGUMENTS_HXX
#define _ARGUMENTS_HXX

#include <map>
#include <string>
#include <vector>

class Arguments {

public:

  Arguments(int argc, char ** argv);

  int Get(const char *name, int defaultValue);
  double Get(const char *name, double defaultValue);
  const char * Get(const char *name, const char * defaultValue);

  bool options_contains(const char *);

private:

  std::map<std::string, std::string> _arguments;
  std::vector<std::string> _options;

};


#endif
