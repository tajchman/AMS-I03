#! /usr/bin/env python

import os, sys, subprocess, argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', default='Release', 
                    choices=['Release', 'Debug'])
parser.add_argument('rest', nargs=argparse.REMAINDER)
args = parser.parse_args()

base = os.path.join('.', 'install', args.type)

resultsDir = os.path.join('.', 'results', args.type)
if not os.path.exists(resultsDir):
   os.makedirs(resultsDir)

code = os.path.join(base, 'PoissonCuda.exe')

subprocess.call([code, "path=" + resultsDir] + args.rest)


