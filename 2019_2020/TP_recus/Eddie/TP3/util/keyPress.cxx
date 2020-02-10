#include <cstdio>
#include <termios.h>
#include <iostream>
#include "keyPress.hxx"

void keyPress(const char *s)
{
  std::cerr << "\n" << s << " ";

  termios oldt, newt;
  int st = fileno(stdin);
  tcgetattr(st, &oldt);

  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO );      
  tcsetattr( st, TCSANOW, &newt);

  getchar();

  tcsetattr( st, TCSANOW, &oldt);
  std::cerr << "continue\n\n";
}
