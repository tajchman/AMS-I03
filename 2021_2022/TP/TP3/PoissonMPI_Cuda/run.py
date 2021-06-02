#! /usr/bin/env python

import os, sys, subprocess, argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nprocs', type=int, default=1)
parser.add_argument('-t', '--type', default='Release', 
                    choices=['Release', 'Debug'])
parser.add_argument('rest', nargs=argparse.REMAINDER)
args = parser.parse_args()

base = os.path.join('.', 'install', args.type)

code = os.path.join(base, 'PoissonMPI_Cuda')
cmd = ['mpiexec', '-n', str(args.nprocs), code] + args.rest

subprocess.call(cmd)


