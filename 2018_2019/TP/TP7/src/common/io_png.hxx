#ifndef __IO_PNG_HXX
#define __IO_PNG_HXX

#define PNG_DEBUG 3
#include <png.h>
#include <vector>


class cImage {
 public:
  cImage() : height(0), width(0), ncolors(0), color_type(0), bit_depth(0) {}
  void resize(int w, int h, int nc) {
    height = h;
    width  = w;
    ncolors = nc;
    coef.resize(h*w*nc);
    if (nc == 1)
      color_type = PNG_COLOR_TYPE_GRAY;
    else if (nc == 3)
      color_type = PNG_COLOR_TYPE_RGB;
      
  }

  float & operator()(int i, int j, int c) {
    return coef[ncolors*(j*width+i) + c];
  }
  
  float operator()(int i, int j, int c) const {
    return coef[ncolors*(j*width+i) + c];
  }
  
  int height, width, ncolors;
  std::vector<float> coef;
  png_byte color_type;
  png_byte bit_depth;

  void read_png(const char *filename);
  void write_png(const char *filename);
 };

#endif

