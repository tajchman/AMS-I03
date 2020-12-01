#! /usr/bin/env python

import os, sys, subprocess, argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', default='Release', 
                    choices=['Debug', 'Release', 'RelWithDebInfo'])
parser.add_argument('-c', '--compilers', default='gnu')
args = parser.parse_args()

myenv = os.environ.copy()

if args.compilers == 'gnu':
    myenv['CC'] = 'gcc'
    myenv['CXX'] = 'g++'
if args.compilers == 'clang':
    myenv['CC'] = 'clang'
    myenv['CXX'] = 'clang++'
elif args.compilers == 'intel':
    myenv['CC'] = 'icc'
    myenv['CXX'] = 'icpc'

for version in ['Seq', 'OpenMP_CoarseGrain']:

  base = os.getcwd()
  srcDir = os.path.join(base, 'src')
  buildDir = os.path.join(base, 'build', version, args.compilers, args.type)
  installDir = os.path.join(base, 'install', args.compilers, args.type)

  if not os.path.exists(buildDir):
    os.makedirs(buildDir)

  if version == "Seq":
    OP = 'false'
  else:
    OP = 'true'

  err = subprocess.call(
    [ 'cmake', 
      '-DCMAKE_BUILD_TYPE=' + args.type,
      '-DCMAKE_INSTALL_PREFIX=' + installDir,
      '-DENABLE_OPENMP='+OP,
      srcDir],
    cwd=buildDir, env=myenv)

  if err == 0:
    err = subprocess.call(
    [ 'make', 
      'install'],
    cwd=buildDir, env=myenv)

