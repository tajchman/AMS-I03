#! /usr/bin/env python

import os, sys, subprocess, argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nprocs', type=int, default=1)
parser.add_argument('-c', '--compilers', default='gnu')
parser.add_argument('rest', nargs=argparse.REMAINDER)
args = parser.parse_args()

base = os.path.join('.', 
                    'install', 
                    args.compilers,
                    'Debug')

resultsDir = os.path.join('.', 'results', args.compilers, 'Debug')
if not os.path.exists(resultsDir):
   os.makedirs(resultsDir)

command = ['mpirun', '-n', str(args.nprocs), 
           'ddd', os.path.join(base, 'PoissonMPI_FineGrain')]

subprocess.call(command + args.rest)


