#! /usr/bin/env python

import os, sys, subprocess, argparse, platform

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', default='Release', 
                    choices=['Debug', 'Release', 'RelWithDebInfo'])
parser.add_argument('-c', '--compilers', default='gnu')
args = parser.parse_args()

myenv = os.environ.copy()
plat = platform.system()

myenv['CUDACXX'] = 'nvcc'

if args.compilers == 'gnu':
  compileCmd = ['make', '--no-print-directory', 'install']
  myenv['CC'] = 'gcc'
  myenv['CXX'] = 'g++'
elif args.compilers == 'clang':
  compileCmd = ['make', '--no-print-directory', 'install']
  myenv['CC'] = 'clang'
  myenv['CXX'] = 'clang++'
elif args.compilers == 'msvc':
  compileCmd = ['ninja', 'install']
  myenv['CC'] = 'cl.exe'
  myenv['CXX'] = 'cl.exe'
elif args.compilers == 'intel':
  if plat == 'Windows':
    compileCmd = ['ninja', 'install']
    myenv['CC'] = 'icl.exe'
    myenv['CXX'] = 'icl.exe'
  else:  
    compileCmd = ['make', '--no-print-directory', 'install']
    myenv['CC'] = 'icc'
    myenv['CXX'] = 'icpc'

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

  if plat == 'Windows':
    cmake_params.append('-GNinja')
  cmake_params.append('-DCMAKE_INSTALL_PREFIX=' + installDir)

  if not os.path.exists(buildDir):
    os.makedirs(buildDir)

  configureCmd = ['cmake'] + cmake_params + [srcDir]
  print(' '.join(configureCmd))

  err = subprocess.call(configureCmd, cwd=buildDir, env=myenv)
  if not err == 0:
    break
  err = subprocess.call(compileCmd, cwd=buildDir, env=myenv)
  if not err == 0:
    break

