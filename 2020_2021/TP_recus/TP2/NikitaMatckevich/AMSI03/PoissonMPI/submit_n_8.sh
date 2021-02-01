#!/bin/bash
#
#  Name of the job (used to build the name of the standard output stream)
#$ -N PoissonMPI_8
#
#  Number of MPI task requested
#$ -pe mpi 8
#
#  The job is located in the current working directory
#$ -cwd
#
#  Merge standard error and standard output streams
#$ -j y

mpirun -np $NSLOTS --display-map /home/matckevich/AMSI03/PoissonMPI/install/Release/PoissonMPI  ;
