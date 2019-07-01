#! /bin/bash

DIR=$( dirname "${BASH_SOURCE[0]}" )
BASE_DIR="$( cd $DIR && pwd )"

if [ "x$MODE" == "x" ]
then
    MODE=Debug
fi

SRC_DIR=$BASE_DIR/src
BUILD_DIR=${BASE_DIR}/build/${MODE}
INSTALL_DIR=${BASE_DIR}/install/${MODE}

NPROCS=1
which nproc >& /dev/null&& NPROCS=`nproc --all`
if [ $NPROCS -gt 10 ] 
then
   NPROCS=10
fi

mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake -G"Eclipse CDT4 - Unix Makefiles" -DCMAKE_BUILD_TYPE=${MODE} \
	-DCMAKE_INSTALL_PREFIX=$INSTALL_DIR $SRC_DIR
make install
