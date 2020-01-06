#! /bin/bash

# Script d'installation de kokkos 
# Adapter les 2 lignes suivantes 
# (repertoires des sources et des binaires)
 
SOURCE_DIR=/opt/AMS_I03/src/Kokkos
INSTALL_DIR=/opt/AMS_I03/Kokkos

mkdir -p $SOURCE_DIR 
cd $SOURCE_DIR

if [ ! -d kokkos ]
then
    git clone https://github.com/kokkos/kokkos.git
fi

for b in OPENMP CUDA
do
    if [ "$b" == "CUDA" ]
    then
       export CXX=$SOURCE_DIR/kokkos/bin/nvcc_wrapper
       FLAGS=-DKOKKOS_ENABLE_CUDA_LAMBDA:BOOL=ON
    fi
    echo CXX = $CXX
    mkdir -p $SOURCE_DIR/build_${b}; cd $SOURCE_DIR/build_${b}
    cmake $FLAGS \
	-DKOKKOS_ENABLE_${b}=ON \
	-DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/kokkos_${b} \
	$SOURCE_DIR/kokkos || exit -1
    
    make  || exit -2
    make install || exit -3
done

