#! /usr/bin/env python

import os, sys, subprocess, platform

myenv = os.environ.copy()
myenv['CUDACXX'] = 'nvcc'

p = platform.system()
if p == 'Windows':
  gen = '-GNinja'
  myenv['CC'] = 'icl.exe'
  myenv['CXX'] = 'icl.exe'
  compileCmd = ['ninja', 'install']
elif p == 'Linux':
  gend = '-GUnix Makefiles'
  myenv['CC'] = 'gcc'
  myenv['CXX'] = 'g++'
  compileCmd = ['make', '--no-print-directory', 'install']

base = os.getcwd()

srcDir = os.path.join(base, 'src')
buildDir = os.path.join(base, 'build')
installDir = os.path.join(base, 'install')

cmake_params = []
if 'CUDA_BIN_DIR' in myenv:
   cmake_params.append('-DCUDA_TOOLKIT_ROOT_DIR=' + myenv['CUDA_BIN_DIR'])
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


