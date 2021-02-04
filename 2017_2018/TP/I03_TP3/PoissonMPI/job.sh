#!/bin/bash
#
#$ -N PoissonMPI
#$ -pe orte 4
#$ -cwd
#$ -j y
#
mpirun --map-by core -display-map ./PoissonMPI  n=400 m=400 p=400

