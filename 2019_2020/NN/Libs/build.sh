#! /bin/bash

DIR=$( dirname "${BASH_SOURCE[0]}" )
BASE_DIR="$( cd $DIR && pwd )"

SRC_DIR=$BASE_DIR/src
BUILD_DIR=${BASE_DIR}/build
INSTALL_DIR=${BASE_DIR}/install

NPROCS=1
which nproc >& /dev/null&& NPROCS=`nproc --all`
if [ $NPROCS -gt 10 ] 
then
   NPROCS=10
fi

for lib in zlib-1.2.11 libpng-1.6.37  
do

mkdir -p $BUILD_DIR/$lib
cd $BUILD_DIR/$lib

cmake \
      -DZLIB_INCLUDE_DIR=$INSTALL_DIR/include \
      -DZLIB_LIBRARY_RELEASE=$INSTALL_DIR/lib/libz.so \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR $SRC_DIR/$lib
make install

done
