#! /bin/bash

\rm -rf results*

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ../src
make
cd -

# perf stat -B -d \
time   ./build/multithreads_2 $@
 
