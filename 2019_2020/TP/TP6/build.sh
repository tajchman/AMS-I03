#! /bin/bash

DIR=`pwd`

NPROCS=1
which nproc >& /dev/null&& NPROCS=`nproc --all`
if [ $NPROCS -gt 10 ] 
then
   NPROCS=10
fi

MODE=Release
mkdir -p $DIR/build
cd $DIR/build

export CC=gcc
export CXX=g++

cmake -DCMAKE_BUILD_TYPE=${MODE} -DCMAKE_INSTALL_PREFIX=$DIR/install $DIR/src
make -j install
