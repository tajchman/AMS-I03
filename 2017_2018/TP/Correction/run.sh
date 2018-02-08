#! /bin/bash

./Poisson_MPI_OpenMP_FineGrain/build/Release/Poisson_MPI threads=2 $@ >& log_$$

