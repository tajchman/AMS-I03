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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', default='Release', 
                    choices=['Debug', 'Release', 'RelWithDebInfo'])
parser.add_argument('-c', '--compiler', default='Gnu', 
                    choices=['Gnu', 'Intel', 'Clang', 'MSVC'])
args = parser.parse_args()

base = os.getcwd()
srcDir = os.path.join(base, 'src')

for o in ["ON", "OFF"]:
  compileCmd, gen, e = envCompiler(args.compiler)

  print ('\nbuild ', args.compiler, args.mode, 'openmp='+o, '\n')
  buildDir = os.path.join(base, 'build', args.compiler, args.mode, o)
  installDir = os.path.join(base, 'install', args.compiler, args.mode)

  cmake_params = ['-DCMAKE_BUILD_TYPE=' + args.mode]
  cmake_params.append('-DCMAKE_INSTALL_PREFIX=' + installDir)
  cmake_params.append('-DENABLE_OPENMP=' + o)
  cmake_params.append(gen)

  if not os.path.exists(buildDir):
    os.makedirs(buildDir)

  configureCmd = ['cmake'] + cmake_params + [srcDir]
  err = subprocess.call(configureCmd, cwd=buildDir, env=e)
  if err == 0:
    err = subprocess.call(compileCmd, cwd=buildDir, env=e)
  if not err == 0:
    exit(-1)

