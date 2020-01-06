

#include <cmath>
#include <iostream>
#include <fstream>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iomanip>

#ifdef _MSC_VER

#include <stdlib.h>
#define bswap_32(x) _byteswap_ulong(x)
#define bswap_64(x) _byteswap_uint64(x)

#elif defined(__APPLE__)

// Mac OS X / Darwin features
#include <libkern/OSByteOrder.h>
#define bswap_32(x) OSSwapInt32(x)
#define bswap_64(x) OSSwapInt64(x)

#elif defined(__sun) || defined(sun)

#include <sys/byteorder.h>
#define bswap_32(x) BSWAP_32(x)
#define bswap_64(x) BSWAP_64(x)

#elif defined(__FreeBSD__)

#include <sys/endian.h>
#define bswap_32(x) bswap32(x)
#define bswap_64(x) bswap64(x)

#elif defined(__OpenBSD__)

#include <sys/types.h>
#define bswap_32(x) swap32(x)
#define bswap_64(x) swap64(x)

#elif defined(__NetBSD__)

#include <sys/types.h>
#include <machine/bswap.h>
#if defined(__BSWAP_RENAME) && !defined(__bswap_32)
#define bswap_32(x) bswap32(x)
#define bswap_64(x) bswap64(x)
#endif

#else

#include <byteswap.h>

#endif

#ifdef _WIN32
constexpr char sep = '\\';
#else
constexpr char sep = '/';   
#endif

struct ImageViewer {

  int nx, ny;
  char * buffer;
  unsigned char * buf2;
  int label;
  int nItems;
  int current;
  std::ifstream fData, fLabel;

  Display * _d;
  XImage * _image;
  GC _gc;
  Visual *_v;
  Window _win;
  int f;
  
  ImageViewer(Display *d, const char * dataName, const char * labelName) :
    _d(d),
    fData(dataName, std::wifstream::in | std::wifstream::binary),
    fLabel(labelName, std::wifstream::in | std::wifstream::binary)
  {
    f = 1;
    int MagicNumber(0);
    fData.read((char *)&MagicNumber, 4);
    MagicNumber = bswap_32(MagicNumber);
    fData.read((char *)&nItems, 4);
    nItems = bswap_32(nItems);
    fData.read((char *)&nx, 4);
    nx = bswap_32(nx);
    fData.read((char *)&ny, 4);
    ny = bswap_32(ny);
    current = -1;
    
    buf2 = new unsigned char[nx * ny];
    buffer = new char[nx*ny];

    int s = DefaultScreen(_d);
    _win = XCreateSimpleWindow(_d, RootWindow(_d, s), 0, 0, nx, ny, 1,
                              BlackPixel(_d, s), WhitePixel(_d, s));
    _gc = XCreateGC(_d, _win, 0, NULL);
    XSelectInput(_d, _win, ExposureMask | KeyPressMask);
    XMapWindow(_d, _win);

    Visual *v = DefaultVisual(d, 0);
    int depth = DefaultDepth(d, s);
    _image = XCreateImage(d, v, depth, ZPixmap, 0, (char *) buffer,
                         nx, ny, 8, 0);
  }

  void Draw(int i) {
    if (i != current) {
      current = i;
      fData.read((char *) buf2, nx * ny);
      for (int p=0; p<nx*ny; p++)
        buffer[p] = (char) buf2[p];

      unsigned char *s;
      int j,k;
      for (k=0, s = buf2; k<ny; k++) {
        for (j=0; j<nx; j++, s++)
          std::cout << " " << std::setw(3) << (int) *s;
        std::cout << std::endl;
      }
    }
    
    std::cout << "Draw" << std::endl;
    XPutImage(_d, _win, _gc, _image, 0, 0, 0, 0, f*nx, f*ny);
  }

  void reDraw()
  {
  }
};


std::string getPathName(const std::string& s) {

  size_t i = s.rfind(sep, s.length());
  if (i != std::string::npos) {
    return(s.substr(0, i+1));
  }

  return("./");
}

int main(int argc, char **argv) {

  // std::string FileData = getPathName(argv[0]) + sep + argv[1];
  // std::string FileLabel = getPathName(argv[0]) + sep + argv[2];

  int item = 0;   
  Display *d;
  XEvent e;
  std::string m;
  int s, i, label;
  int depth;
  
  d = XOpenDisplay(NULL);
  if (d == NULL) {
    fprintf(stderr, "Cannot open display\n");
    exit(1);
  }
   
  ImageViewer Image(d,
                    "train-images.idx3-ubyte",
                    "train-labels.idx1-ubyte");
  i = 0;
  std::string msg = std::to_string(i) + ": ";
  while (1) {
    XNextEvent(d, &e);
    if (e.type == Expose) {
      Image.Draw(i);
    //   XFillRectangle(d, w, DefaultGC(d, s), 20, 20, 10, 10);
    //  XDrawString(d, w, DefaultGC(d, s), 10, 50, msg, strlen(msg));
    }
    if (e.type == KeyPress)
      break;
  }
 
  XCloseDisplay(d);

  return 0;
}
