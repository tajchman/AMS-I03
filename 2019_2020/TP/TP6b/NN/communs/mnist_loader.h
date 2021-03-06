#pragma once

#include "boost/numeric/ublas/vector.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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

using namespace boost::numeric;

// Loads the MNIST data files
template <typename T> class mnist_loader {
public:
  mnist_loader(const std::string &FileData,
               const std::string &FileLabels,
               std::vector<std::pair<ublas::vector<T>,
               ublas::vector<T>>> &mnist_data) {
    {
      std::ifstream myFile(FileData, std::wifstream::in | std::wifstream::binary);
      if (!myFile)
        throw "File does not exist";
      int MagicNumber(0);
      int nItems(0);
      int nRows(0);
      int nCol(0);
      myFile.read((char *)&MagicNumber, 4);
      MagicNumber = bswap_32(MagicNumber);
      if (MagicNumber != 2051)
        throw "Magic number for training data incorrect";
      myFile.read((char *)&nItems, 4);
      nItems = bswap_32(nItems);
      myFile.read((char *)&nRows, 4);
      nRows = bswap_32(nRows);
      myFile.read((char *)&nCol, 4);
      nCol = bswap_32(nCol);
      std::unique_ptr<unsigned char[]> buf(new unsigned char[nRows * nCol]);
      for (auto i = 0; i < nItems; ++i) {
        myFile.read((char *)buf.get(), nRows * nCol);
        ublas::vector<T> data(nRows * nCol);
        for (auto j = 0; j < nRows * nCol; ++j) {
          data[j] = static_cast<T>(buf[j]) / static_cast<T>(255.0);
        }
        mnist_data.push_back(make_pair(data, ublas::zero_vector<T>(10)));
      }
    }
    {
      std::ifstream myFile(FileLabels,
                           std::wifstream::in | std::wifstream::binary);
      if (!myFile)
        throw "File does not exist";
      int MagicNumber(0);
      int nItems(0);
      myFile.read((char *)&MagicNumber, 4);
      MagicNumber = bswap_32(MagicNumber);
      if (MagicNumber != 2049)
        throw "Magic number for label file incorrect";
      myFile.read((char *)&nItems, 4);
      nItems = bswap_32(nItems);
      for (int i = 0; i < nItems; ++i) {
        char data;
        myFile.read(&data, 1);
        mnist_data[i].second[data] = 1.0;
      }
    }
  }
};
