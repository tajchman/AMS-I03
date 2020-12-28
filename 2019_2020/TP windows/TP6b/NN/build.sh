#! /bin/bash

DIR=`pwd`
for i in ${DIR}/NN* ${DIR}/Data
do
  cd $i
  ./build.sh
done

