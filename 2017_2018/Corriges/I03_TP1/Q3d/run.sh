#! /bin/bash

\rm -rf results*

mkdir -p build
cd build
cmake ../src
make VERBOSE=1
cd -

perf stat -B -d \
   ./build/PoissonSeq  n=300 m=300 p=300 it=10
     
