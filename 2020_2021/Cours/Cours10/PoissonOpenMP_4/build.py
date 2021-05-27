#! /usr/bin/env python

import os, sys, subprocess, argparse, platform

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', default='Release', 
                    choices=['Debug', 'Release', 'RelWithDebInfo'])
args = parser.parse_args()

myenv = os.environ.copy()
p = platform.system()
if p == 'Windows':
  gen = '-GNinja'
  if 'CC' not in myenv:
    myenv['CC'] = 'icl.exe'
    myenv['CXX'] = 'icl.exe'
  compileCmd = ['ninja', 'install']
elif p == 'Linux':
  gen = '-GUnix Makefiles'
  if 'CC' not in myenv:
      myenv['CC'] = 'gcc'
      myenv['CXX'] = 'c++'
  compileCmd = ['make', '--no-print-directory', 'install']

base = os.getcwd()
srcDir = os.path.join(base, 'src')


for v in ["ON", "OFF"]:
  print ('\nbuild ', args.type, ' OpenMP:', v, '\n')
  buildDir = os.path.join(base, 'build', v, args.type)
  installDir = os.path.join(base, 'install', args.type)

  cmake_params = ['-DCMAKE_BUILD_TYPE=' + args.type]
  cmake_params.append('-DCMAKE_INSTALL_PREFIX=' + installDir)
  cmake_params.append('-DENABLE_OPENMP=' + v)
  cmake_params.append(gen)

  if not os.path.exists(buildDir):
    os.makedirs(buildDir)

  configureCmd = ['cmake'] + cmake_params + [srcDir]
  print(' '.join(configureCmd))
  err = subprocess.call(configureCmd, cwd=buildDir, env=myenv)
  if err == 0:
    err = subprocess.call(compileCmd, cwd=buildDir, env=myenv)
  if not err == 0:
    break

