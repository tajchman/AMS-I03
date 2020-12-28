#ifndef __OS_HXX__
#define __OS_HXX__

#include <cstdlib>

#include <stdio.h>  /* defines FILENAME_MAX */
#ifdef _WIN32
#include <direct.h>
#define currentDir _getcwd
   constexpr char sep = '\\';
#else
#include <unistd.h>
#define currentDir getcwd
   constexpr char sep = '/';   
#endif
#include<iostream>
 
std::string GetCurrentDir( void ) {
  char buff[FILENAME_MAX];
  currentDir( buff, FILENAME_MAX );
  std::string current_working_dir(buff);
  return current_working_dir;
}
 
std::string GetDirName(const std::string& s, int n = 1) {

  std::string t = s;
  if (t.size() > 2 && t[0] == '.' && t[1] == sep)
    t = GetCurrentDir() + sep + t.substr(2);
  std::cerr << t << std::endl;
  
  for (; n--; ) {
   size_t i = t.rfind(sep, t.length());
   if (i != std::string::npos)
     t = t.substr(0, i);
   else
     t = ".";
  }
  return t;
}

#endif
