/*
 * input.cxx
 *
 *  Created on: 7 mars 2019
 *      Author: marc
 */

#include "input.hxx"

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdint>
#include <cstddef>
#include "io_png.hxx"

inline void littleBigEndian (uint32_t &x) {
  x = ((x >> 24) & 0xffL)
    | ((x >> 8) & 0xff00L)
    | ((x << 8) & 0xff0000L)
    | ((x << 24) & 0xff000000L);
}

struct stream {
  
  stream(const char *fileName, uint32_t magicNumber)
    : file(fileName, std::ios::binary) {
    uint32_t m = readInt32();
    if (! m == magicNumber)
      throw std::invalid_argument("not a labels file");
    std::cout << "\nmagic number     : " <<  m << std::endl;
    n = readInt32();
  }
  
  uint32_t readInt32() {
    unsigned char i[4], k;
    
    file.read((char*) i, 4);
    k = i[0]; i[0] = i[3]; i[3] = k;
    k = i[1]; i[1] = i[2]; i[2] = k;
    return * (const uint32_t *) i;
  }
  unsigned char readByte() {
    unsigned char b;
    file.read((char*) &b, 1);
    return b;
  }
  void readBytes(std::vector<unsigned char> &v) {
    file.read((char*) v.data(), v.size());
  }
  std::size_t n;
  std::ifstream file;
};

struct labelStream {
  labelStream (const char *fileName) : f(fileName, 2049) {
    std::cout << "number of labels : " << f.n << std::endl;
  }
  
  size_t n() const {
    return f.n;
  }
  
  bool next(unsigned char & v) {
    v = f.readByte();
    return true;
  }
  
  stream f;
};

struct imageStream {
  imageStream (const char *fileName) : f(fileName, 2051) {
    std::cout << "number of images : " << f.n << std::endl;
    nrows = f.readInt32();
    ncols = f.readInt32();
    std::cout << "image size       : " << nrows
	      << " x " << ncols << std::endl;
  }
  
  size_t n() const {
    return f.n;
  }
  
  bool next(std::vector<unsigned char> & coefs) {
    if (! (coefs.size() == nrows*ncols))
      coefs.resize(nrows*ncols);
    f.readBytes(coefs);
    return true;
  }
  
  size_t nrows, ncols;
  stream f;
};

input::input(const char *labelName, const char *imageName)
{
  vbuffer = 0;
  labelFile = labelName ? new labelStream(labelName) : NULL;
  imageFile = imageName ? new imageStream(imageName) : NULL;
  if (imageFile && labelFile)
    if (! (labelFile->n() == imageFile->n()))
      throw  std::invalid_argument("not the same sizes of label file and image file");
  m_n = imageFile ? imageFile->ncols * imageFile->nrows : 0;
  k = 0;
}

input::~input()
{
  if (labelFile) delete labelFile;
  if (imageFile) delete imageFile;
}

bool input::next(vector &p, double &v)
{
  size_t n = imageFile->ncols * imageFile->nrows;
  bool ok;

  if (p.size() < n)
    p.resize(n);
  if (buffer.size() < n)
    buffer.resize(n);

  ok = labelFile->next(vbuffer);
  if (ok)
    v = vbuffer;

  ok = ok && imageFile->next(buffer);
  if (ok)
    for (size_t i=0; i<n; i++) p[i] = buffer[i];

  k=k+1;
  if (k < 100) {
    cImage I( int(imageFile->ncols),  int(imageFile->nrows));
    I.coef = buffer;
    
    std::ostringstream s;
	s << "train_" << k << ".png";
    
    I.write_png(s.str().c_str());
  }
  
  return ok;
}
