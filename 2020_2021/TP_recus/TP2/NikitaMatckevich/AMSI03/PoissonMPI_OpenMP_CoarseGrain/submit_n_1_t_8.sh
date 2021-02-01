#!/bin/bash
#
#  Name of the job (used to build the name of the standard output stream)
#$ -N output_n_8_t_1
#
#  Number of MPI task requested
#$ -pe mpi 8
#
#  The job is located in the current working directory
#$ -cwd
#
#  Merge standard error and standard output streams
#$ -j y

mpirun -np 1 --map-by socket --display-map /home/matckevich/AMSI03/PoissonMPI_OpenMP_CoarseGrain/install/R/PoissonMPI_CoarseGrain threads=8  ;
