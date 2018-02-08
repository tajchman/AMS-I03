#! /bin/bash

SUPPR="--suppressions=/opt/intel/tools/install/share/openmpi/openmpi-valgrind.supp"

valgrind ${SUPPR} ./Poisson_MPI/build/Poisson_MPI n=20 m=20 p=20 it=2 >& log_$$

