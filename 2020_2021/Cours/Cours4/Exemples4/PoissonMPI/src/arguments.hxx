#ifndef _ARGUMENTS_HXX
#define _ARGUMENTS_HXX

#include <map>
#include <string>
#include <set>


enum TypeArgument {
  TypeInt,
  TypeReal,
  TypeString,
  TypeBool,
};

struct UnknownArgument : public std::exception
{
  UnknownArgument(const char * name) {
    _message = "argument inconnu : <";
    _message += name;
    _message += ">";
  }
  const char * what()
  {
     return _message.c_str();
  }
  std::string _message;
};

struct UnknownOption : public std::exception
{
  UnknownOption(const char * name) {
    _message = "option inconnue : <";
    _message += name;
    _message += ">";
  }
  const char * what()
  {
     return _message.c_str();
  }
  std::string _message;
};

struct BadTypeArgument : public std::exception
{
  BadTypeArgument(const char * name, const char *value, TypeArgument type) {
    _message = "erreur de type sur la valeur de l'argument : <";
    _message += name;
    _message += ">, valeur : ";
    _message += value;
    _message += ", type attendu : ";
    if (type == TypeInt)
      _message += "entier";
    else if (type == TypeReal)
      _message += "reel";
    else if (type == TypeString)
      _message += "chaine de caracteres";
    else if (type == TypeBool)
      _message += "[vrai/faux] [0/1] [true/false]";
  }
  const char * what()
  {
     return _message.c_str();
  }
  std::string _message;
};

struct Any {
   int intVal;
   double doubleVal;
   bool boolVal;
   std::string stringVal;

   Any() {}

   void Set(int n) { intVal = n; } 
   void Set(bool b) { boolVal = b; }
   void Set(double x) { doubleVal = x; } 
   void Set(const std::string &s) { stringVal = s; }

   void Get(int & n) { n = intVal; }
   void Get(bool & n) { n = boolVal; }
   void Get(double & n) { n = doubleVal; }
   void Get(std::string & n) { n = stringVal; }
};
struct Argument {
  std::string name;
  TypeArgument type;
  Any value;
};

class Arguments {

public:

  Arguments() {
    _options["-h"] = false;
    _options["--help"] = false;
  }
  void AddOption(const char *name);
  template<typename T>
  void AddArgument(const char * name, T defaultValue);

  void Parse(int argc, char ** argv);

  void Usage();

  template<typename T>
  void Get(const char *name, T & value) {
    auto t = _arguments.find(name);
    if (t == _arguments.end())
     throw UnknownArgument(name);

    t->second.value.Get(value);
  };
 
  template<typename T>
  void Set(const char *name, const T & value) {
    auto t = _arguments.find(name);
    if (t == _arguments.end())
      throw UnknownArgument(name);

    t->second.value.Set(value);
  }

  bool GetOption(const char *s) {
    if (_options.find(s) == _options.end())
      throw UnknownOption(s);

    return _options[s];
  }

private:

  std::string _cmd;
  std::map<std::string, Argument> _arguments;
  std::map<std::string, bool> _options;

};


#endif
