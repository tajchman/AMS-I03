#! /usr/bin/env python

import os, sys, subprocess, argparse, signal

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nprocs', type=int, default=1)
parser.add_argument('-c', '--compilers', default='gnu')
parser.add_argument('-t', '--type', default='Release', 
                    choices=['Release', 'Debug'])
parser.add_argument('rest', nargs=argparse.REMAINDER)
args = parser.parse_args()

base = os.path.join(os.getcwd(), 'install', args.compilers, args.type)
code = os.path.join(base, 'PoissonMPI_FineGrain.exe')

resultsDir = os.path.join(os.getcwd(), 'results', args.compilers, args.type)
if not os.path.exists(resultsDir):
   os.makedirs(resultsDir)

command = ['mpirun', '-n', str(args.nprocs), '--xterm', '-1!', code]


def get_pid(name):
    try:
        l = map(int, subprocess.check_output(["pidof",name]).split())
    except:
        l = []
    return l

pid_before = get_pid('xterm')

try:
    print("Taper control-C pour arrêter ... ")
    subprocess.call(command + args.rest, cwd=resultsDir)
except KeyboardInterrupt:
    pid_after = get_pid('xterm')
    for p in pid_after:
        if not p in pid_before:
            os.kill(p, signal.SIGINT)





