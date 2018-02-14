#! /bin/bash

if [ "x$version" == "x" ] 
then
	version=Poisson_MPI_OpenMP_CoarseGrain
fi

ddd ./${version}/build/Debug/${version} &

