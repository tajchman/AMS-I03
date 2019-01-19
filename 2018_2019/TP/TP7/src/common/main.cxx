#include "timer.hxx"
#include "io_png.hxx"
#include "operation.h"
#include <stddef.h>
#include <iostream>
#include <string>

int main(int argc, char **argv)
{
  Timer T_global;
  T_global.start();
  
  std::string fileIn, fileOut;
  cImage imageIn, imageOut;
  
  fileIn = argc > 1 ? argv[1] : "install/ecureuil.png";
  fileOut = argc > 2 ? argv[2] : fileIn;
  std::size_t found = fileOut.rfind("/");
  if (found != std::string::npos && found < fileOut.size()-1)
    fileOut = fileOut.insert(found+1, "res_"); 

  Timer T1, T2, T3;
  T1.start();
  
  std::cerr << "\nIn : " << fileIn << std::endl;
  imageIn.read_png(fileIn.c_str());
  imageIn.write_png("test.png");
  
  T1.stop();
  std::cerr << "\n\tTime read file  " << T1.elapsed() << " s" << std::endl;
  
  T2.start();
  
  process(imageOut, imageIn);
  
  T2.stop();
  std::cerr << "\n\tTime processing " << T2.elapsed() << " s" << std::endl;
  
  T3.start();

  std::cerr << "\nOut : " << fileOut << std::endl;
  imageOut.write_png(fileOut.c_str());
  
  T3.stop();
  std::cerr << "\n\tTime write file " << T3.elapsed() << " s" << std::endl;

  T_global.stop();
  std::cerr << "\n\tTime (total)    " << T_global.elapsed() << " s" << std::endl;
  return 0;
}
