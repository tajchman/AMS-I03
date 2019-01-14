#include "io_png.hxx"

cImage rgb_to_grey(cImage &imageIn)
{
  int n =imageIn.height * imageIn.width, i;
  cImage imageOut(imageIn.height, imageIn.width);
  
  for (i=0; i<n; i++)
    imageOut.c[i] = imageIn.c[i*4];

  return imageOut;
}
