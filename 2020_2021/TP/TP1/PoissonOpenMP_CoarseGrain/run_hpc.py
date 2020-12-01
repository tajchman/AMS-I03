#! /usr/bin/env python

import os, subprocess, shutil, glob

baseDir = os.getcwd()
BASETK='/opt/hpctoolkit/spack/opt/spack/linux-ubuntu20.10-zen2/gcc-10.2.0'

HPCBASE=os.path.join(BASETK, 'hpctoolkit-2020.08.03-s65cel4htgfwauk3d72vv3tlkez6fjcj/bin')
HPCTRACE=os.path.join(BASETK, 'hpcviewer-2020.07-6euoi7nyf4pxla2trwtdjoev4jqerdk6/bin')

e = os.environ.copy()
e['PATH'] = ':'.join([HPCBASE, HPCTRACE, e['PATH']])

compilateur = 'gnu'
build = 'RelWithDebInfo'
version = 'PoissonOpenMP_FineGrain'

code = os.path.join(baseDir, 'install', compilateur, build, version)

testDir=os.path.join(baseDir, 'analyse', 'hpctoolkit')

if os.path.exists(testDir):
   shutil.rmtree(testDir)
   
os.makedirs(testDir)

for f in glob.glob('hpctoolkit-*') + glob.glob(version + '*'):
   if os.path.isdir(f):
      shutil.rmtree(f)
   else:
      os.remove(f)


subprocess.call(['hpcstruct', code], env=e, cwd=testDir)
subprocess.call(['hpcrun', code, 'threads=4'], env=e, cwd=testDir)
subprocess.call(['hpcprof', '-S', version + '.hpcstruct', 'hpctoolkit-' + version + '-measurements'], env=e, cwd=testDir)

subprocess.Popen(['hpctraceviewer', 'hpctoolkit-' + version + '-database'], env=e, cwd=testDir)
subprocess.Popen(['hpcviewer', 'hpctoolkit-' + version + '-database'], env=e, cwd=testDir)
