#include "arguments.hxx"
#include <cstring>
#include <iostream>

void Arguments::AddOption(const char *name)
{
  _options[name] = false;
}

template <>
void Arguments::AddArgument(const char * name, int defaultValue)
{
  Argument A;
  A.name = name;
  A.value.Set(defaultValue);
  A.type = TypeInt;
  _arguments[name] = A;
}
template <>
void Arguments::AddArgument(const char * name, bool defaultValue)
{
  Argument A;
  A.name = name;
  A.value.Set(defaultValue);
  A.type = TypeBool;
  _arguments[name] = A;
}

template <>
void Arguments::AddArgument(const char * name, double defaultValue)
{
  Argument A;
  A.name = name;
  A.value.Set(defaultValue);
  A.type = TypeReal;
  _arguments[name] = A;
}

template <>
void Arguments::AddArgument(const char * name, const char * defaultValue)
{
  Argument A;
  A.name = name;
  A.value.Set(defaultValue);
  A.type = TypeString;
  _arguments[name] = A;
}

void Arguments::Parse(int argc, char **argv)
{
  _cmd = argv[0];

  for (int iarg = 1; iarg < argc; iarg++) {
    const char *s = argv[iarg];

    if (strlen(s) > 1) {
      if (s[0] == '-') {
        if (_options.find(s) != _options.end())
          _options[s] = true;
      }
      else {
        const char * p = strchr(s, '=');
        if (p) {
          std::string k(s, p);
          const char *key = k.c_str();
          std::string v(p+1);
          const char *value = v.c_str();
          auto a = _arguments.find(key);
          if (a != _arguments.end()) {
            auto & b = a->second;
            char *end;
            if (b.type == TypeInt) {
              b.value.intVal = strtol(value, &end, 10);
              if (*end != '\0')
                throw BadTypeArgument(key, value, TypeInt);
            }
            else if (b.type == TypeReal) {
              b.value.doubleVal = strtod(value, &end);
              if (*end != '\0')
                throw BadTypeArgument(key, value, TypeReal);
            }
            else if (b.type == TypeString)
              b.value.stringVal = value;
            else if (b.type == TypeBool) {
              if (strcasecmp(value, "true") == 0 || 
                  strcmp(value, "1") == 0 || 
                  strcasecmp(value, "vrai") == 0)
                b.value.boolVal = true;
              else if (strcasecmp(value, "false") == 0 || 
                  strcmp(value, "0") == 0 ||
                  strcasecmp(value, "faux") == 0)
                b.value.boolVal = false;
              else
                throw BadTypeArgument(key, value, TypeBool);
            }           
          }
        }
      }
    }
  }
}

void Arguments::Usage()
{
  std::cout << "Usage : " << _cmd;

  for (auto & o : _options)
    std::cout << " [" << o.first << "]";

  for (auto & a : _arguments) {
    std::cout << " [" << a.first << "=";
    if (a.second.type == TypeInt)
       std::cout << "<int>";
    if (a.second.type == TypeReal)
       std::cout << "<real>";
    if (a.second.type == TypeString)
       std::cout << "<string>";
    if (a.second.type == TypeBool)
       std::cout << "<0/1/false/true/faux/vrai>";
    std::cout << "]";
  }
  std::cout << std::endl;
}



