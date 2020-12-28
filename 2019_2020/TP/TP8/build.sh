#! /bin/bash

DIR=`pwd`

NPROCS=1
which nproc >& /dev/null&& NPROCS=`nproc --all`
if [ $NPROCS -gt 10 ] 
then
   NPROCS=10
fi


CMAKE_FLAGS=" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
CMAKE_FLAGS+=" -DCMAKE_BUILD_TYPE=Debug"
#CMAKE_FLAGS+=" -DCMAKE_BUILD_TYPE=Release"

for d in C++11 Sequentiel TBB PyCuda
do
    printf "\n\t%30s\n\n" $d

    mkdir -p $DIR/build/$d
    cd $DIR/build/$d

    echo "cmake ${CMAKE_FLAGS} -DCMAKE_INSTALL_PREFIX=$DIR/install $DIR/src/$d"

    cmake ${CMAKE_FLAGS} \
       -DCMAKE_INSTALL_PREFIX=$DIR/install \
        $DIR/src/$d || exit -1
    make || exit -2
    make install || exit -3
    
done

for k in OPENMP CUDA
do
    d=kokkos_$k
    printf "\n\t%30s\n\n" $d

    mkdir -p $DIR/build/$d
    cd $DIR/build/$d

    echo "cmake ${CMAKE_FLAGS} -DKOKKOS_ENABLE_$k=ON -DCMAKE_INSTALL_PREFIX=$DIR/install $DIR/src/Kokkos"

    echo k=$k
    export KOKKOS_ROOT=${KOKKOS_BASE}/$d
    if [ "$k" == "CUDA" ]
    then
       export CXX=$KOKKOS_ROOT/bin/nvcc_wrapper
    fi
    echo $CXX

    cmake ${CMAKE_FLAGS} \
       -DKOKKOS_ENABLE_$k=ON \
       -DCMAKE_INSTALL_PREFIX=$DIR/install \
        $DIR/src/Kokkos || exit -1
    make || exit -2
    make install || exit -3
    
done

