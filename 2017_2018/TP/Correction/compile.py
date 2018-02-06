#! /usr/bin/env python

import os, sys, subprocess

DIR=os.getcwd();

if sys.platform.startswith('win'):
   shell = True
else:
   shell = False

for i in ('Sequential', 'OpenMP_FineGrain', 'OpenMP_CoarseGrain', 'MPI'):
   sys.stderr.write(i + '\n')
   BUILD_DIR=os.path.join(DIR, 'Poisson' + i, 'build')
   if not os.path.exists(BUILD_DIR):
      os.makedirs(BUILD_DIR)   
   os.chdir(BUILD_DIR)

   e = os.environ.copy()

   if i == 'MPI':
     e['CC']  = 'mpicc-mpich-mp'
     e['CXX'] = 'mpicxx-mpich-mp'
   else:
     e['CC']  = 'gcc-mp-7'
     e['CXX'] = 'g++-mp-7'
     
   retCode = subprocess.check_call(['cmake', '../src'],
                                   env=e,
                                   stderr=subprocess.STDOUT,
                                   shell=shell)
  
   retCode = subprocess.check_call(['make'],
                                   stderr=subprocess.STDOUT,
                                   shell=shell)
  

