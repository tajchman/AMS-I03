#! /bin/bash

DIR=`pwd`
for i in ${DIR}/NN*
do
  cd $i
  ./build.sh
done

