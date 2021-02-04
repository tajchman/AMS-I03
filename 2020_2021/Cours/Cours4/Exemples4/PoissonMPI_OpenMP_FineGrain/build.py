#! /usr/bin/env python

import os, sys, subprocess, argparse, platform

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', default='Release', 
                    choices=['Debug', 'Release', 'RelWithDebInfo'])
parser.add_argument('-c', '--compilers', default='gnu')
args = parser.parse_args()

myenv = os.environ.copy()
plat = platform.system()

compileCmd = ['make', '--no-print-directory', 'install']
myenv['CC'] = 'mpicc'
myenv['CXX'] = 'mpicxx'

base = os.getcwd()
srcDir = os.path.join(base, 'src')
buildDir = os.path.join(base, 'build')
installDir = os.path.join(base, 'install')

cmake_params = ['-DCMAKE_BUILD_TYPE=' + args.type]
cmake_params.append('-DCMAKE_INSTALL_PREFIX=' + installDir)

if not os.path.exists(buildDir):
  os.makedirs(buildDir)

configureCmd = ['cmake'] + cmake_params + [srcDir]
print(' '.join(configureCmd))
err = subprocess.call(configureCmd, cwd=buildDir, env=myenv)

if err == 0:
  err = subprocess.call(compileCmd, cwd=buildDir, env=myenv)

