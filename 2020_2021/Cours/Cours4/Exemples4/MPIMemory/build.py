#! /usr/bin/env python

import os, sys, subprocess, argparse, platform

parser = argparse.ArgumentParser()
args = parser.parse_args()

myenv = os.environ.copy()
plat = platform.system()

compileCmd = ['make', '--no-print-directory', 'install']
myenv['CC'] = 'mpicc'
myenv['CXX'] = 'mpicxx'

base = os.getcwd()
srcDir = os.path.join(base, 'mpimemu-1.3')
buildDir = os.path.join(base, 'build')
installDir = os.path.join(base, 'install')

if not os.path.exists(buildDir):
  os.makedirs(buildDir)

configureCmd = [os.path.join(srcDir, 'configure'), '--prefix=' + installDir]
print(' '.join(configureCmd))
err = subprocess.call(configureCmd, cwd=buildDir, env=myenv)

if err == 0:
  err = subprocess.call(compileCmd, cwd=buildDir, env=myenv)

