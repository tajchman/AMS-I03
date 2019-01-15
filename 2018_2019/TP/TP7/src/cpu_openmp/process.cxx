#include "io_png.hxx"

cImage process(cImage &imageIn)
{
  int i, j, k, c;
  
  cImage imageOut(imageIn.width, imageIn.height);
  imageOut.color_type = imageIn.color_type;
  imageOut.bit_depth = imageIn.bit_depth;

#pragma omp parallel for private(i,j,c)
  for (i=1; i<imageIn.width-1; i++) {
    for (j=1; j<imageIn.height-1; j++) {
      for (c=0; c<3; c++)
	imageOut(i,j,c) = 0.5 + (- 4 * imageIn(i,j,c)
				 + imageIn(i+1,j+1,c)
				 + imageIn(i-1,j-1,c)
				 + imageIn(i-1,j+1,c)
				 + imageIn(i+1,j-1,c));
    }
  }
  
  return imageOut;
}
