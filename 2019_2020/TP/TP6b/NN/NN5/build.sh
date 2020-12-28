#! /bin/bash

DIR=`pwd`

for b in Debug Release
do
  mkdir -p ${DIR}/build_${b}
  cd ${DIR}/build_${b}
  cmake -DCMAKE_BUILD_TYPE=${b} -DCMAKE_INSTALL_PREFIX=${DIR}/install_${b} ${DIR}/src
  make && make install
done
