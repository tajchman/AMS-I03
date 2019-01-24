#! /bin/bash

DIR=`pwd`

NPROCS=1
which nproc >& /dev/null&& NPROCS=`nproc --all`
if [ $NPROCS -gt 10 ] 
then
   NPROCS=10
fi


CMAKE_FLAGS=" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
#CMAKE_FLAGS+=" -DCMAKE_BUILD_TYPE=Debug"
CMAKE_FLAGS+=" -DCMAKE_BUILD_TYPE=Release"

case "x$OSTYPE" in
    xdarwin*)
        export CC=clang
        export CXX=clang++
        ;;
    xlinux-gnu*)
        if [ "x$CUDA_GCC" == "x" ]
        then
            export CC=gcc
            export CXX=g++       
        else
            export CC=$CUDA_GCC
            export CXX=${CUDA_GCC/%gcc/g++}
            CMAKE_FLAGS+=" -DCMAKE_CUDA_HOST_COMPILER=${CXX}"
        fi
        ;; 
esac

mkdir -p $DIR/build
cd $DIR/build
ln -sf $DIR/src 
echo "cmake ${CMAKE_FLAGS} -DCMAKE_INSTALL_PREFIX=$DIR/install $DIR/src"
cmake ${CMAKE_FLAGS} -DCMAKE_INSTALL_PREFIX=$DIR/install $DIR/src
make install
