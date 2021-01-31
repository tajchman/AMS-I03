#include "timer.hxx"
#include "io_png.hxx"
#include "operation.h"
#include <stddef.h>
#include <iostream>
#include <string>
#include "version.hxx"

const char pathSeparator =
#ifdef _WIN32
      '\\';
#else
      '/';
#endif

int main(int argc, char **argv)
{
  Timer T_global;
  T_global.start();
  
  std::string fileIn, fileOut, fileOutTxt;
  cImage imageIn, imageOut;
 
  std::string defaultIn = INSTALL_PREFIX;
  defaultIn += pathSeparator;
  defaultIn += "/ecureuil.png";

  fileIn = argc > 1 ? argv[1] : defaultIn;
  if (argc > 2) {
    fileOut = argv[2];
  }
  else {
    std::string prefix = "res_";
    prefix += version;
    prefix += "_";

    fileOut = fileIn;
    std::size_t found = fileOut.rfind(pathSeparator);
    if (found != std::string::npos && found < fileOut.size()-1)
        fileOut.insert(found+1, prefix); 
  }
  fileOutTxt = fileOut;
  size_t l = fileOutTxt.size();
  fileOutTxt.replace(l-3, 3, "txt");
 
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

  std::cerr << "\nOut : " << fileOut << " text: " << fileOutTxt << std::endl;
  imageOut.write_png(fileOut.c_str());
  imageOut.write_txt(fileOutTxt.c_str());
  
  T3.stop();
  std::cerr << "\n\tTime write file " << T3.elapsed() << " s" << std::endl;

  T_global.stop();
  std::cerr << "\n\tTime (total)    " << T_global.elapsed() << " s" << std::endl;
  return 0;
}
