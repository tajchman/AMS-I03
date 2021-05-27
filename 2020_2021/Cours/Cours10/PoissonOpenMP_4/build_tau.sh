#! /bin/bash

module load perf

DIR=`pwd`
echo $DIR

tau_cxx.sh -optAppCXX="g++-9" -optPdtCxxOpts="-Isrc -Isrc/util --std=c++11" \
    src/main.cxx  \
    src/parameters.cxx \
    src/scheme.cxx \
    src/user.cxx \
    src/values.cxx \
    src/util/arguments.cxx \
    src/util/os.cxx \
    src/util/timer.cxx -o PoissonOpenMP
