#! /bin/bash

\rm -rf results*

mkdir -p build
cd build
cmake ../src
make
cd -

perf stat -B -d \
   ./build/PoissonSeq  n=300 m=300 p=300 it=20
 
