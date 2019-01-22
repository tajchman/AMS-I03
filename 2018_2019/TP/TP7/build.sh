#! /bin/bash

DIR=`pwd`

NPROCS=1
which nproc >& /dev/null&& NPROCS=`nproc --all`
if [ $NPROCS -gt 10 ] 
then
   NPROCS=10
fi

case "x$OSTYPE" in
 xdarwin*)
   export CC=clang
   export CXX=clang++
   ;;
 *)
   export CC=gcc
   export CXX=g++
   ;; 
esac

mkdir -p $DIR/build
cd $DIR/build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$DIR/install $DIR/src
make install
