#! /usr/bin/env python

import os, sys, subprocess
execfile('./arch.py')

args = sys.argv[1:];

BUILD_MODE = []
for arg in args:
    if arg == "Debug":
        BUILD_MODE.append("Debug")
    if arg == "Release":
        BUILD_MODE.append("Release");

if BUILD_MODE == []:
    BUILD_MODE=["Release"]

DIR=os.getcwd();

if sys.platform.startswith('win'):
    shell = True
else:
    shell = False

for b in BUILD_MODE:
    for i in ('Sequential', 'OpenMP_FineGrain', 'OpenMP_CoarseGrain', 'MPI', 'MPI_OpenMP_FineGrain', 'MPI_OpenMP_CoarseGrain'):
        sys.stderr.write(i + '\n')
        SRC_DIR=os.path.join(DIR, 'Poisson_' + i, 'src')
        BUILD_DIR=os.path.join(DIR, 'Poisson_' + i, 'build', b)
        if not os.path.exists(BUILD_DIR):
            os.makedirs(BUILD_DIR)   
        os.chdir(BUILD_DIR)
        
        e = os.environ.copy()

        if i == 'MPI':
            e['CC']  = mMPICC
            e['CXX'] = mMPICXX
            pass
        else:
            e['CC']  = mCC
            e['CXX'] = mCXX
            pass
 
        retCode = subprocess.check_call(['cmake',
                                         '-DCMAKE_BUILD_TYPE=' + b,
                                         SRC_DIR],
                                        env=e,
                                        stderr=subprocess.STDOUT,
                                        shell=shell)
        
        retCode = subprocess.check_call(['make'],
                                        stderr=subprocess.STDOUT,
                                        shell=shell)
        pass
    pass

