#! /bin/bash

rm log*
rm -rf res*

echo "run 1 proc"
mpirun -n 1 ./run.sh $@

echo "run 2 proc"
mpirun -n 2 ./run.sh $@

