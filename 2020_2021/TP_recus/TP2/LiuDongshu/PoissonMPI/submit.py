#! /usr/bin/env python

import os, sys, subprocess, argparse, signal, platform

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nprocs', type=int, default=1)
parser.add_argument('rest', nargs=argparse.REMAINDER)
args = parser.parse_args()

testDir = os.getcwd()

s = '''#!/bin/bash
#
#  Name of the job (used to build the name of the standard output stream)
#$ -N PoissonMPI_{0}
#
#  Number of MPI task requested
#$ -pe mpi {0}
#
#  The job is located in the current working directory
#$ -cwd
#
#  Merge standard error and standard output streams
#$ -j y

mpirun -np $NSLOTS --display-map {1}/install/Release/PoissonMPI {2} ;
'''.format(args.nprocs, testDir, ' '.join(args.rest))

subFile = 'submit_n_' + str(args.nprocs) + '.sh'
with open(subFile, 'w') as f: 
   f.write(s)

subprocess.call(['qsub', subFile])
