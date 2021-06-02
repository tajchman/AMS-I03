#! /usr/bin/env python

import os, subprocess, shutil, argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n0', type=int,default=401)
parser.add_argument('-n1', type=int,default=401)
parser.add_argument('-n2', type=int,default=401)

args = parser.parse_args()

options = ['n0=' + str(args.n0), 'n1=' + str(args.n1), 'n2=' + str(args.n2)]

for v in ["Seq", "Cuda"]:
  resDir = 'res_' + v

  if os.path.exists(resDir):
    shutil.rmtree(resDir)
  os.makedirs(resDir)
  
  execDir = os.path.join('.', 'Poisson' + v, 'install', 'Release')
  code = os.path.join(execDir, 'Poisson' + v + '.exe')
  cmd = [code] + options + ["path=" + resDir]
  print(" ".join(cmd))
  
  with open(os.path.join(resDir, "out"), "w") as f:
    subprocess.run(cmd, stdout=f)

#subprocess.run(['meld', 'res_Seq', 'res_Cuda'])
