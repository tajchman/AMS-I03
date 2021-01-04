#include <cstdio>
#include <iostream>
#include <limits>

void keypress(const  char *s)
  {
  std::cout << s << " ... ";
  std::cin.ignore( std::numeric_limits <std::streamsize> ::max(), '\n' );
  std::cout << " continue" << std::endl;
  }
