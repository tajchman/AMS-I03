#include "arguments.hxx"
#include <cstring>
#include <iostream>

Arguments::Arguments(int argc, char **argv)
{
  for (int iarg = 1; iarg < argc; iarg++) {
    const char *s = argv[iarg];

    if (strlen(s) > 1) {
      if (s[0] == '-') {
        s++;
        if (s[0] == '-')
           _options.push_back(s + 1);
      }
      else {
        const char * p = strchr(s, '=');
        if (p) {
          std::string key(s, p);
          std::string value(p+1);
           _arguments[key] = value;
        }
      }
    }
  }
}

bool Arguments::options_contains(const char *s) {

   bool v = false;
   for (auto & e : _options)
     if (e == s) {
       v = true;
       break;
     }

   return v;
}

int Arguments::Get(const char * name, int defaultValue)
{
  auto t = _arguments.find(name);
  if (t == _arguments.end())
     return defaultValue;
  return strtol(t->second.c_str(), NULL, 10);
}

double Arguments::Get(const char *name, double defaultValue)
{
  auto t = _arguments.find(name);
  if (t == _arguments.end())
     return defaultValue;
  return strtod(t->second.c_str(), NULL);
}
  
const char * Arguments::Get(const char *name, const char * defaultValue)
{
  auto t = _arguments.find(name);
  if (t == _arguments.end())
     return defaultValue;
  return t->second.c_str();
}
