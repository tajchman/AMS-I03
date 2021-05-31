#! /usr/bin/env python

import subprocess, os, sys, argparse, glob
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', default='Release', 
                    choices=['Debug', 'Release', 'RelWithDebInfo'])
parser.add_argument('-c', '--compiler', default='gnu',
                    choices=['gnu', 'intel'])
args = parser.parse_args()

listd = glob.glob('v*')
listd.sort()

for d in listd:
    e = os.environ.copy()
    if args.compiler == 'gnu':
        e['CC'] = 'gcc'
        e['CXX'] = 'g++'
    elif args.compiler == 'intel':
        e['CC'] = 'icc'
        e['CXX'] = 'icpc'

    if d in ['v6', 'v7']:
        e['OMP_NUM_THREADS'] = '3'
    command = ['python', 'build.py', '--type', args.type]
    subprocess.run(command, cwd=d, env=e)
    
