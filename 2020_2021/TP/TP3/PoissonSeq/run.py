#! /usr/bin/env python

import os, sys, subprocess, argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', default='Release', 
                    choices=['Release', 'Debug'])
parser.add_argument('rest', nargs=argparse.REMAINDER)
args = parser.parse_args()

base = os.path.join('.', 
                    'install',
                    args.type)

codeSeq = os.path.join(base, 'PoissonSeq')
print(codeSeq)
subprocess.call([codeSeq] + args.rest)


