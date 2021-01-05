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
#$ -N myjob
#
#  Number of MPI task requested
#$ -pe mpi {}
#
#  The job is located in the current working directory
#$ -cwd
#
#  Merge standard error and standard output streams
#$ -j y

source {}/env.sh
mpirun -map-by slot:pe={} -np {} --mca btl vader,self --display-map {}/install/Release/PoissonMPI {} ;
'''.format(args.nprocs * args.nthreads, testDir, args.nthreads, args.nprocs, testDir, ' '.join(args.rest))

print s
