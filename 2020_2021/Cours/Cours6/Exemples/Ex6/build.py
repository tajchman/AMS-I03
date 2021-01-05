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
  myenv['CC'] = 'gcc'
  myenv['CXX'] = 'g++'
  compileCmd = ['make', '--no-print-directory', 'install']

base = os.getcwd()

for d in [
         'addition_CPU',
         'addition_CPU_OpenMP',
         'addition_GPU',
         'addition_GPU_reduction',
         'Pi_CPU',
         'Pi_GPU'
         ]:
  srcDir = os.path.join(base, 'src', d)
  buildDir = os.path.join(base, 'build', d)
  installDir = os.path.join(base, 'install')

  cmake_params = ['-DCMAKE_BUILD_TYPE=' + args.type]
  cmake_params.append('-DCMAKE_INSTALL_PREFIX=' + installDir)
  cmake_params.append(gen)

  if not os.path.exists(buildDir):
    os.makedirs(buildDir)

  configureCmd = ['cmake'] + cmake_params + [srcDir]
  err = subprocess.call(configureCmd, cwd=buildDir, env=myenv)
  if not err == 0:
    break
  err = subprocess.call(compileCmd, cwd=buildDir, env=myenv)
  if not err == 0:
    break

