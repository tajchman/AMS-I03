#ifndef __IO_PNG_HXX
#define __IO_PNG_HXX

#define PNG_DEBUG 3
#include <png.h>
#include <cstdlib>


class cImage {
 public:
  cImage(int w, int h) : height(h), width(w),
			 coef((float *) malloc(sizeof(float)*h*w*3)) {
  }

  float & operator()(int i, int j, int c) {
    return coef[3*(j*width+i) + c];
  }
  
  int height, width;
  float *coef;
  png_byte color_type;
  png_byte bit_depth;

  void clean() { free(coef); height = 0; width = 0; }
 };

cImage read_png_file (const char *filename);
void write_png_file(const char *filename, cImage & I);

#endif

