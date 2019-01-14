#include "io_png.hxx"

cImage rgb_to_blue(cImage imageIn, int h, int w)
{
  int n =imageIn.height * imageOut.width, i;
  cImage imageOut;
  
  for (i=0; i<n; i++)
    imageOut.c[i] = imageIn.c[i*4];

  return imageOut;
}
