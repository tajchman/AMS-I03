#! /bin/bash

if [ "x$version" == "x" ] 
then
	version=Poisson_MPI_OpenMP_CoarseGrain
fi

./${version}/build/Release/${version} $@ >& log_$$

