#! /usr/bin/env python

import os, sys, subprocess, argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', default='Release', 
                    choices=['Debug', 'Release'])
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

base = os.getcwd()
srcDir = os.path.join(base, 'src')
buildDir = os.path.join(base, 'build', args.compilers)
installDir = os.path.join(base, 'install', args.compilers)

if not os.path.exists(buildDir):
  os.makedirs(buildDir)

err = subprocess.call(
  ['cmake', 
   '-DCMAKE_BUILD_TYPE=' + args.type,
   '-DCMAKE_INSTALL_PREFIX=' + installDir,
  srcDir],
    cwd=buildDir, env=myenv)

if err == 0:
  err = subprocess.call(
    ['make', 
     'install'],
    cwd=buildDir, env=myenv)

