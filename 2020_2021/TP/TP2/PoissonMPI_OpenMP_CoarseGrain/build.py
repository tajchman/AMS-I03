#! /usr/bin/env python

import os, sys, subprocess, argparse, platform

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', default='Release', 
                    choices=['Debug', 'Release', 'RelWithDebInfo'])
args = parser.parse_args()

myenv = os.environ.copy()
myenv['CC'] = 'mpicc'
myenv['CXX'] = 'mpicxx'
compileCmd = ['make', '--no-print-directory', 'install']

base = os.getcwd()
srcDir = os.path.join(base, 'src')

for t in args.type:
  print ('\nbuild ', t, '\n')
  buildDir = os.path.join(base, 'build', t)
  installDir = os.path.join(base, 'install', t)

  cmake_params = ['-DCMAKE_BUILD_TYPE=' + t]
  cmake_params.append('-DCMAKE_INSTALL_PREFIX=' + installDir)

  if not os.path.exists(buildDir):
    os.makedirs(buildDir)

  configureCmd = ['cmake'] + cmake_params + [srcDir]
  err = subprocess.call(configureCmd, cwd=buildDir, env=myenv)

  if err == 0:
    err = subprocess.call(compileCmd, cwd=buildDir, env=myenv)

