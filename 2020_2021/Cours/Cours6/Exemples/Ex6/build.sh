#! /bin/bash

DIR=`pwd`

MODE=Release
mkdir -p $DIR/build
cd $DIR/build

export CC=gcc
export CXX=g++

cmake -G Ninja -DCMAKE_BUILD_TYPE=${MODE} -DCMAKE_INSTALL_PREFIX=$DIR/install $DIR/src
make -j install
