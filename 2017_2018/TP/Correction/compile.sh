#! /bin/bash

DIR=`pwd`

for i in Sequential OpenMP_FineGrain OpenMP_CoarseGrain MPI
do
   cd Poisson_$i
   mkdir -p build
   cd build
   cmake ../src
   make
   cd $DIR
done

