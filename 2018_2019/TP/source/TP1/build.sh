#! /bin/bash

DIR=`pwd`

module load gnu 

mkdir -p $DIR/build
cd $DIR/build
cmake -G "Eclipse CDT4 - Unix Makefiles" ../src
make
