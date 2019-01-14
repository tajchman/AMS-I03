#include "io_png.hxx"
#include "operation.h"
#include <stddef.h>

int main(char **argv, int argc)
{
  char *fileIn, *fileOut;
  cImage imageIn, imageOut;

  imageIn = read_png_file (char *filename);
  
  fileIn = argc > 1 ? argv[1] : "install/ecureuil.png";
  fileOut = argc > 2 ? argv[2] : "new_ecureuil.png";

  imageIn = read_png_file (fileIn);

  imageOut = rgb_to_grey(imageIn);
  
  write_png_file(fileOut, imageOut);

  return 0;
}
