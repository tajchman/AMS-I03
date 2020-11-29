#! /usr/bin/env python

import os, sys, subprocess, argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', default='Release', 
                    choices=['Debug', 'Release'])
args = parser.parse_args()

base = os.getcwd()
srcDir = os.path.join(base, 'src')
buildDir = os.path.join(base, 'build', args.type)
installDir = os.path.join(base, 'install')

if not os.path.exists(buildDir):
  os.makedirs(buildDir)

err = subprocess.call(
  ['cmake', 
   '-DCMAKE_BUILD_TYPE=' + args.type,
   '-DCMAKE_INSTALL_PREFIX=' + installDir,
  srcDir],
    cwd=buildDir)

if err == 0:
  err = subprocess.call(
    ['make', 
     'install'],
    cwd=buildDir)

