#include "io_png.hxx"

cImage rgb_to_grey(cImage &imageIn)
{
  int i, j;
  
  cImage imageOut(imageIn.width, imageIn.height);
  imageOut.color_type = imageIn.color_type;
  imageOut.bit_depth = imageIn.bit_depth;

  for (i=0; i<imageIn.width; i++) {
    for (j=0; j<imageIn.height; j++) {
      imageOut(i,j,0) = imageIn(i,j,0);
      imageOut(i,j,1) = imageIn(i,j,1);
      imageOut(i,j,2) = 0;
    }
  }
  
  return imageOut;
}
