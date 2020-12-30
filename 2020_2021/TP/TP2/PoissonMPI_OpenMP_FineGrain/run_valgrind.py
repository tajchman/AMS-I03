#! /usr/bin/env python

import os, sys, subprocess, argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nprocs', type=int, default=1)
parser.add_argument('-c', '--compilers', default='gnu')
parser.add_argument('-t', '--type', default='Release', 
                    choices=['Release', 'Debug'])
parser.add_argument('rest', nargs=argparse.REMAINDER)
args = parser.parse_args()

base = os.path.join('.', 
                    'install', 
                    args.compilers,
                    args.type)

resultsDir = os.path.join('.', 'results', args.compilers, args.type)
if not os.path.exists(resultsDir):
   os.makedirs(resultsDir)

command = ['mpirun', '-n', str(args.nprocs), 
           '--xterm', '-1!', 'valgrind', 
           '--leak-check=full', '--show-leak-kinds=all',
           '--suppressions=/usr/share/openmpi/openmpi-valgrind.supp',
           os.path.join(base, 'PoissonMPI.exe')]

subprocess.call(command + args.rest)


