#ifndef __IO_PNG_HXX
#define __IO_PNG_HXX

#define PNG_DEBUG 3
#include <png.h>
#include <cstdlib>


class cImage {
 public:
  cImage(int h, int w) : height(h), width(w),
			 c((float *) malloc(sizeof(float)*h*w*4)) {}
  int height, width;
  float *c;
  png_byte color_type;
  png_byte bit_depth;

  void clean() { free(c); height = 0; width = 0; }
 };

cImage read_png_file (const char *filename);
void write_png_file(const char *filename, cImage & I);

#endif

