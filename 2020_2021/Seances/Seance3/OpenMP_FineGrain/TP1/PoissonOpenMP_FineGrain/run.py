#! /usr/bin/env python
# -*- coding: utf-8 -*-

try:
    unicode('')
except NameError:
    unicode = str

import os, sys, argparse
from subprocess import *

parser = argparse.ArgumentParser()
parser.add_argument('threadsMax', type=int)
parser.add_argument('-c', '--compilers', default='gnu')
parser.add_argument('-t', '--type', default='Release', 
                    choices=['Release', 'Debug'])
parser.add_argument('rest', nargs=argparse.REMAINDER)
args = parser.parse_args()

version = 'OpenMP_FineGrain'
base = os.path.join('.', 
                    'install', 
                    args.compilers,
                    args.type)

resultsDir = os.path.join('.', 'results', args.compilers, args.type)
if not os.path.exists(resultsDir):
   os.makedirs(resultsDir)

codeSeq = os.path.join(base, 'PoissonSeq.exe')
codePar = os.path.join(base, 'Poisson' + version + '.exe')

t = []
x = []
last = []
speedup = []

def readTemps(n):
    s = os.path.join(resultsDir, 'temps_' + str(n) + '.dat')
    with open(s) as f:
        l = f.readline().split()
        return (float(l[1]), float(l[2]))
    return (0.0, 0.0)

with open('run_' + version + '.log', 'w') as log:
    
    proc = Popen([codeSeq, "path=" + resultsDir] + args.rest,
                 stdout=PIPE, encoding='utf-8')
    while proc.poll() is None:
        text = proc.stdout.readline() 
        log.write(text)
        sys.stdout.write(text)
        
    x0,v0 = readTemps(0)

    t = []
    x = []
    last = []
    speedup = []

    for i in range(1,args.threadsMax+1):
       proc = Popen([codePar, 'threads=' + str(i), 'path=' + resultsDir]
                   + args.rest, stdout=PIPE, encoding='utf-8')
       while proc.poll() is None:
           text = proc.stdout.readline() 
           log.write(text)
           sys.stdout.write(text)

       (u,v) = readTemps(i)
       t.append(i)
       x.append(u)
       last.append(v)
       speedup.append(x0/u)
            

with open('run_' + version + '.log', 'a') as log:
    s = 'last diff (sequential) = ' + "{:12.3f}".format(v0) + "\n\n"
    sys.stdout.write(s)
    log.write(s)

    s = "threads:  " +  "".join(["{:12d}".format(u) for u in t]) + "\n"
    sys.stdout.write(s)
    log.write(s)
    
    s = "speedups: " + "".join(["{:12.3f}".format(u) for u in speedup]) + "\n"
    sys.stdout.write(s)
    log.write(s)
    
    s = "last diff:" + "".join(["{:12.3f}".format(u) for u in last]) + "\n"
    sys.stdout.write(s)
    log.write(s)
    
import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
    display = False
else:
    display = True

import matplotlib.pyplot as plt


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,16))

ax1.plot(t, x, 'o-')
ax1.set_ylabel('Temps CPU (s)')
ax1.xaxis.set_ticks(range(1,args.threadsMax+1))
ax1.grid()

ax2.plot(t, speedup, 'o-')
ax2.plot(t, t, '-')
plt.ylim(0, args.threadsMax)
ax2.legend([unicode('mesure'), unicode('ideal')])
ax2.set_xlabel('Threads')
ax2.set_ylabel('Speedup')
ax2.xaxis.set_ticks(range(1,args.threadsMax+1))
ax2.grid()

fig.set_size_inches(6, 7)
plt.savefig("speedups_OpenMP_CoarseGrain.pdf")

try:
   if display:
      plt.show()
except:
    pass



