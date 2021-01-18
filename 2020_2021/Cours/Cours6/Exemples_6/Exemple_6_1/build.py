#! /usr/bin/env python

import os, sys, subprocess, argparse, platform

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', default='Release', 
                    choices=['Debug', 'Release', 'RelWithDebInfo'])
args = parser.parse_args()

myenv = os.environ.copy()
myenv['CUDACXX'] = 'nvcc'

p = platform.system()
if p == 'Windows':
  gen = '-GNinja'
  myenv['CC'] = 'icl.exe'
  myenv['CXX'] = 'icl.exe'
  compileCmd = ['ninja', 'install']
elif p == 'Linux':
  gen = '-GUnix Makefiles'
  myenv['CC'] = 'gcc'
  myenv['CXX'] = 'g++'
  compileCmd = ['make', '--no-print-directory', 'install']
elif p == 'Darwin':
  gen = '-GUnix Makefiles'
  myenv['CC'] = 'gcc-10'
  myenv['CXX'] = 'g++-10'
  compileCmd = ['make', '--no-print-directory', 'install']

base = os.getcwd()

srcDir = os.path.join(base, 'src')

t = args.type
print ('\nbuild ', t, '\n')
buildDir = os.path.join(base, 'build', t)
installDir = os.path.join(base, 'install', t)

cmake_params = ['-DCMAKE_BUILD_TYPE=' + t]
cmake_params.append('-DCMAKE_INSTALL_PREFIX=' + installDir)
cmake_params.append(gen)

if not os.path.exists(buildDir):
  os.makedirs(buildDir)

configureCmd = ['cmake'] + cmake_params + [srcDir]
err = subprocess.call(configureCmd, cwd=buildDir, env=myenv)
if not err == 0:
  sys.exit(-1)
err = subprocess.call(compileCmd, cwd=buildDir, env=myenv)
if not err == 0:
  sys.exit(-1)


