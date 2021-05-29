#! /usr/bin/env python

import subprocess, os, sys, argparse, glob

listd = glob.glob('v*')
listd.sort()

exe = os.path.join('install', 'Release', 'bin', 'nbody')

for d in listd:
    e = os.environ.copy()
    if d in ['v6', 'v7']:
        e['OMP_NUM_THREADS'] = '3'
    command = ['python', 'build.py']
    subprocess.run(command, cwd=d, env=e)
    
