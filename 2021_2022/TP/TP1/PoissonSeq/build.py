#! /usr/bin/env python

import os, sys, subprocess, argparse, platform

def envCompiler(compiler):
  print(compiler)
  p = platform.system()
  e = os.environ.copy()
  if p == 'Windows':
    gen = '-GNinja'
    if compiler == 'Intel':
       e['CC'] = 'icl.exe'
       e['CXX'] = 'icl.exe'
    elif compiler == 'MSVC':
       e['CC'] = 'cl.exe'
       e['CXX'] = 'cl.exe'
    else:
      raise 'Compiler type must be Intel or MSVC'
    compileCmd = ['ninja', 'install']
  elif p == 'Linux':
    gen = '-GUnix Makefiles'
    compileCmd = ['make', '--no-print-directory', 'install']
    if compiler == 'Gnu':
       e['CC'] = 'gcc'
       e['CXX'] = 'g++'
    elif compiler == 'Intel':
       e['CC'] = 'icc'
       e['CXX'] = 'icpc'
    elif compiler == 'Clang':
       e['CC'] = 'clang'
       e['CXX'] = 'clang++'
    else:
      raise 'Compiler type must be Intel, Gnu or Clang'

  return compileCmd, gen, e

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', default='Release', 
                    choices=['Debug', 'Release', 'RelWithDebInfo'])
parser.add_argument('-c', '--compiler', default='Gnu', 
                    choices=['Gnu', 'Intel', 'Clang', 'MSVC'])
args = parser.parse_args()

compileCmd, gen, e = envCompiler(args.compiler)

base = os.getcwd()
srcDir = os.path.join(base, 'src')

print ('\nbuild ', args.compiler, args.type, '\n')
buildDir = os.path.join(base, 'build', args.compiler, args.type)
installDir = os.path.join(base, 'install', args.compiler, args.type)

cmake_params = ['-DCMAKE_BUILD_TYPE=' + args.type]
cmake_params.append('-DCMAKE_INSTALL_PREFIX=' + installDir)
cmake_params.append(gen)

if not os.path.exists(buildDir):
  os.makedirs(buildDir)

configureCmd = ['cmake'] + cmake_params + [srcDir]
err = subprocess.call(configureCmd, cwd=buildDir, env=e)
if err == 0:
  err = subprocess.call(compileCmd, cwd=buildDir, env=e)

