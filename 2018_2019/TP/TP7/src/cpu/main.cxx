#include "io_png.hxx"
#include "operation.h"
#include <stddef.h>

int main(int argc, char **argv)
{
  const char *fileIn, *fileOut;
  fileIn = argc > 1 ? argv[1] : "install/ecureuil.png";
  fileOut = argc > 2 ? argv[2] : "new_ecureuil.png";

  cImage imageIn = read_png_file (fileIn);

  cImage imageOut = rgb_to_grey(imageIn);
  
  write_png_file(fileOut, imageOut);

  return 0;
}
