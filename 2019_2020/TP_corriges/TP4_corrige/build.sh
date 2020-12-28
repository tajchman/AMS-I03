#! /bin/bash

DIR=`pwd`

for i in PoissonSeq PoissonOpenMP_FineGrain PoissonOpenMP_CoarseGrain
do
  cd $i
  ./build.sh
  cd $DIR
done

