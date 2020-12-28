#! /bin/bash

DIR=`pwd`

NPROCS=1
which nproc >& /dev/null&& NPROCS=`nproc --all`
if [ $NPROCS -gt 10 ] 
then
   NPROCS=10
fi

  mkdir -p $DIR/build
  cd $DIR/build
  cmake -G "Eclipse CDT4 - Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$DIR/install $DIR/src
  make -j
  make install
