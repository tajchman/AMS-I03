#include "timer.hxx"
#include "io_png.hxx"
#include "operation.h"
#include <stddef.h>
#include <iostream>

int main(int argc, char **argv)
{
  const char *fileIn, *fileOut;
  fileIn = argc > 1 ? argv[1] : "install/ecureuil.png";
  fileOut = argc > 2 ? argv[2] : "new_ecureuil.png";

  Timer T1, T2, T3;
  T1.start();
  
  cImage imageIn = read_png_file (fileIn);
  
  T1.stop();
  std::cerr << "Time read     " << T1.elapsed() << std::endl;
  
  T2.start();
  
  cImage imageOut = process(imageIn);
  
  T2.stop();
  std::cerr << "Time process " << T2.elapsed() << std::endl;
  
  T3.start();
  
  write_png_file(fileOut, imageOut);
  
  T3.stop();
  std::cerr << "Time write " << T3.elapsed() << std::endl;

  return 0;
}
