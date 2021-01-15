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
 
  std::string defaultIn = INSTALL_PREFIX;
  defaultIn += "/ecureuil.png";

  fileIn = argc > 1 ? argv[1] : defaultIn;
  if (argc > 2) {
    fileOut = argv[2];
  }
  else {
    fileOut = fileIn;
    std::size_t found = fileOut.rfind("/");
    if (found != std::string::npos && found < fileOut.size()-1)
    fileOut.insert(found+1, "res_"); 
  }
  
  Timer T1, T2, T3;
  T1.start();
  
  std::cerr << "\nIn : " << fileIn << std::endl;
  imageIn.read_png(fileIn.c_str());
  
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
