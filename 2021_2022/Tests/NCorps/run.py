#! /usr/bin/env python

import subprocess, os, sys, argparse, glob
parser = argparse.ArgumentParser()

parser.add_argument('-t', '--type', default='Release', 
                    choices=['Debug', 'Release', 'RelWithDebInfo'])
parser.add_argument('-c', '--compiler', default='Gnu',
                    choices=['Gnu', 'Intel', 'Clang', 'MSVC'])
parser.add_argument('-p', '--particles', type=int, default = 50000)
parser.add_argument('versions', nargs='*', default = ['*'])
args = parser.parse_args()

if args.versions == ['*']:
    listd = glob.glob('v*')
    listd.sort()
else:
    listd = ['v' + v for v in args.versions]


exe = os.path.join('install', args.compiler, args.type, 'bin', 'nbody')

for d in listd:
    e = os.environ.copy()
    if d in ['v6', 'v7']:
        e['OMP_NUM_THREADS'] = '3'
    command = [os.path.join(d, exe), str(args.particles)]
    subprocess.run(command, env=e)
    
[]