#! /usr/bin/env python

import os, sys, subprocess, argparse, signal, platform

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nprocs', type=int, default=1)
parser.add_argument('-t', '--nthreads', type=int, default=1)
parser.add_argument('rest', nargs=argparse.REMAINDER)
args = parser.parse_args()

testDir = os.getcwd()

s = '''#!/bin/bash
#
#  Name of the job (used to build the name of the standard output stream)
#$ -N output_n_{0}_t_{1}
#
#  Number of MPI task requested
#$ -pe mpi {2}
#
#  The job is located in the current working directory
#$ -cwd
#
#  Merge standard error and standard output streams
#$ -j y

mpirun -np {1} --map-by socket --display-map {3}/install/Release/PoissonMPI_FineGrain threads={0} {4} ;
'''.format(args.nthreads, args.nprocs, args.nprocs * args.nthreads, testDir, ' '.join(args.rest))


subFile = 'submit_n_' + str(args.nprocs) + '_t_' + str(args.nthreads) + '.sh'
with open(subFile, 'w') as f:
   f.write(s)

subprocess.call(['qsub', subFile])
