#!/bin/bash
#
#  Name of the job (used to build the name of the standard output stream)
#$ -N output_n_2_t_4
#
#  Number of MPI task requested
#$ -pe mpi 8
#
#  The job is located in the current working directory
#$ -cwd
#
#  Merge standard error and standard output streams
#$ -j y

mpirun -np 4 --map-by socket --display-map /home/matckevich/AMSI03/PoissonMPI_OpenMP_FineGrain/install/Release/PoissonMPI_FineGrain threads=2  ;
