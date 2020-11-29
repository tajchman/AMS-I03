#! /usr/bin/env python

import os, sys, subprocess, argparse

parser = argparse.ArgumentParser()
parser.add_argument('threadsMax', type=int)
parser.add_argument('-c', '--compilers', default='gnu')
parser.add_argument('-t', '--type', default='Release', 
                    choices=['Release', 'Debug'])
args = parser.parse_args()

base = os.path.join('.', 
                    'install', 
                    args.compilers,
                    args.type)

resultsDir = os.path.join('.', 'results', args.compilers, args.type)

codeSeq = os.path.join(base, 'PoissonSeq')
codePar = os.path.join(base, 'PoissonOpenMP_FineGrain')

subprocess.call([codeSeq, "path=" + resultsDir])
for i in range(1,args.threadsMax+1):
    subprocess.call([codePar, 'threads=' + str(i), "path=" + resultsDir])

with open('./.run.py', 'w') as f:
    f.write('resultsDir = "' + resultsDir + '"\n')
    f.write('threads = ' + str(args.threadsMax) + '\n')


