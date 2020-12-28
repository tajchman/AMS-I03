#! /bin/bash

\rm -rf results*

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ../src
make
cd -

# time \
perf stat -B -d \
./build/sequentiel/compteSeq $@

for t in 1 2
do
# time \
perf stat -B -d \
./build/multithreads_1/compteMT_1 $@ -t $t
done

for t in 1 2
do
# time \
perf stat -B -d \
./build/multithreads_2/compteMT_2 $@ -t $t
done
