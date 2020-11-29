#! /usr/bin/env python

import os, sys, glob, subprocess, argparse

parser = argparse.ArgumentParser()
parser.add_argument('threads', type=int)
args = parser.parse_args()

for i in range(1,args.threads+1):
    subprocess.call(['./install/PoissonOpenMP_CoarseGrain', 'threads=' + str(i)])


