#! /usr/bin/env python

import os, sys, subprocess, argparse, signal, platform

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nprocs', type=int, default=1)
parser.add_argument('rest', nargs=argparse.REMAINDER)
args = parser.parse_args()

p = platform.system()
print(p)
if p == 'Windows':
    compiler = 'Intel'
elif p == 'Linux':
    compiler = 'Gnu'

base = os.path.join(os.getcwd(), 'build', 'Release')
code = os.path.join(base, 'PoissonMPI_FineGrain')

resultsDir = os.path.join(os.getcwd(), 'results', 'Release')
if not os.path.exists(resultsDir):
   os.makedirs(resultsDir)

command = ['mpiexec', '-n', str(args.nprocs), code]


def get_pid(name):
    try:
        l = map(int, subprocess.check_output(["pidof",name]).split())
    except:
        l = []
    return l

pid_before = get_pid('xterm')

try:
    print("Taper control-C pour arreter ... ")
    subprocess.call(command + args.rest, cwd=resultsDir)
except KeyboardInterrupt:
    pid_after = get_pid('xterm')
    for p in pid_after:
        if not p in pid_before:
            os.kill(p, signal.SIGINT)


