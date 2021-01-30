#! /usr/bin/env python3

import os, sys, subprocess, argparse

parser = argparse.ArgumentParser()
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

codeSeq = os.path.join(base, 'PoissonSeq.exe')

subprocess.call([codeSeq, "path=" + resultsDir] + args.rest)


