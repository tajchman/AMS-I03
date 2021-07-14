#! /usr/bin/env python
# -*- coding: utf-8 -*-

try:
    unicode('')
except NameError:
    unicode = str

import os, sys, argparse, numpy
from subprocess import *

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--compilers', default='Gnu')
parser.add_argument('-t', '--type', default='Release', 
                    choices=['Release', 'Debug', 'RelWithDebInfo'])
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

call([codeSeq, "path=" + resultsDir] + args.rest)


