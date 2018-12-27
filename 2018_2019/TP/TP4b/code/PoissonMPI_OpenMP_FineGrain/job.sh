#!/bin/bash
#
#$ -N PoissonHybrid_4x4
#$ -pe orte 4
#$ -cwd
#$ -j y
#
export LD_LIBRARY_PATH=/share/apps/gcc/current/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=4

mpirun -x OMP_NUM_THREADS --map-by core -display-map /home/tajchman/AMSI03/test/AMS_I03/2018_2019/TP/TP4b/code/PoissonMPI_OpenMP_FineGrain/install/PoissonMPI_OpenMP_FineGrain  n=500 m=500 p=500

