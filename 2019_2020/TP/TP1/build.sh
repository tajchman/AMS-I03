#! /bin/bash

DIR=`pwd`

NPROCS=1
which nproc && NPROCS=`nproc --all`
if [ $NPROCS -gt 10 ] 
then
   NPROCS=10
fi

for i in Debug Profile Release
do
  mkdir -p $DIR/build/$i
  cd $DIR/build/$i
  cmake -DCMAKE_BUILD_TYPE=$i -DCMAKE_INSTALL_PREFIX=$DIR/install/$i $DIR/src
  make -j
  make -j install
done

