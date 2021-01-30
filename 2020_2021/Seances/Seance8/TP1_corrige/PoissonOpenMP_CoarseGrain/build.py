#! /usr/bin/env python3

import os, sys, subprocess, argparse, platform

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', default='Release', 
                    choices=['Debug', 'Release', 'RelWithDebInfo'])
parser.add_argument('-c', '--compilers', default='gnu')
args = parser.parse_args()

myenv = os.environ.copy()
plat = platform.system()

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
  myenv['CC'] = 'cl'
  myenv['CXX'] = 'cl'
elif args.compilers == 'intel':
  if plat == 'Windows':
    compileCmd = ['ninja', 'install']
    myenv['CC'] = 'icl.exe'
    myenv['CXX'] = 'icl.exe'
  else:  
    compileCmd = ['make', '--no-print-directory', 'install']
    myenv['CC'] = 'icc'
    myenv['CXX'] = 'icpc'

for version in ['Seq', 'OpenMP_CoarseGrain']:

  base = os.getcwd()
  srcDir = os.path.join(base, 'src')
  buildDir = os.path.join(base, 'build', version, args.compilers, args.type)
  installDir = os.path.join(base, 'install', args.compilers, args.type)

  cmake_params = ['-DCMAKE_BUILD_TYPE=' + args.type]

  if plat == 'Windows':
    cmake_params.append('-GNinja')
  cmake_params.append('-DCMAKE_INSTALL_PREFIX=' + installDir)

  if not os.path.exists(buildDir):
    os.makedirs(buildDir)

  if not version == "Seq":
      cmake_params.append('-DENABLE_OPENMP=ON')

  configureCmd = ['cmake'] + cmake_params + [srcDir]
  print(' '.join(configureCmd))
  err = subprocess.call(configureCmd, cwd=buildDir, env=myenv)

  if err == 0:
      err = subprocess.call(compileCmd, cwd=buildDir, env=myenv)

