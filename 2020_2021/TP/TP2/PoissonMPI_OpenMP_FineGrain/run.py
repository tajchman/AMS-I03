#! /usr/bin/env python

import os, sys, subprocess, argparse, signal

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

command = ['mpirun', '-n', str(args.nprocs), '-report-pid', 'pids', 
           '--xterm', '-1!', os.path.join(base, 'PoissonMPI.exe')]


def get_pid(name):
    try:
        l = map(int, subprocess.check_output(["pidof",name]).split())
    except:
        l = []
    return l

pid_before = get_pid('xterm')

try:
    print("Taper control-C pour arrêter ... ")
    subprocess.call(command + args.rest)
except KeyboardInterrupt:
    pid_after = get_pid('xterm')
    for p in pid_after:
        if not p in pid_before:
            os.kill(p, signal.SIGINT)





