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
parser.add_argument('-c', '--compilers', default='Gnu')
parser.add_argument('-t', '--type', default='Release', 
                    choices=['Release', 'Debug', 'RelWithDebInfo'])
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

def readTemps(n):
    s = os.path.join(resultsDir, 'temps_' + str(n) + '.dat')

    with open(s) as f:
        l = f.readline().split()
        return (float(l[1]), float(l[2]), float(l[3]))
    return (0.0, 0.0, 0.0)

with open('run_' + version + '.log', 'w') as log:
    
    proc = Popen([codeSeq, "path=" + resultsDir] + args.rest,
                 stdout=PIPE, encoding='utf-8')
    while proc.poll() is None:
        text = proc.stdout.readline() 
        log.write(text)
        sys.stdout.write(text)
        
    t_total_0, t_transitoire_0, last_diff_0 = readTemps(0)
    threads = numpy.zeros(args.threadsMax, dtype=int)
    t_total = numpy.zeros(args.threadsMax)
    t_transitoire = numpy.zeros(args.threadsMax)
    last = numpy.zeros(args.threadsMax)
    speedup = numpy.zeros(args.threadsMax)
    speedup_transitoire = numpy.zeros(args.threadsMax)
    efficiency = numpy.zeros(args.threadsMax)
    efficiency_transitoire = numpy.zeros(args.threadsMax)

    for i in range(1,args.threadsMax+1):
       proc = Popen([codePar, 'threads=' + str(i), 'path=' + resultsDir]
                   + args.rest, stdout=PIPE, encoding='utf-8')
       while proc.poll() is None:
           text = proc.stdout.readline() 
           log.write(text)
           sys.stdout.write(text)

       t_total_i, t_transitoire_i, last_diff_i = r = readTemps(i)
       threads[i-1] = i
       t_total[i-1] = t_total_i
       t_transitoire[i-1] = t_transitoire_i
       last[i-1] = last_diff_i
       speedup[i-1] = t_total_0/t_total_i
       speedup_transitoire[i-1] = t_transitoire_0/t_transitoire_i
       efficiency[i-1] = t_total_0/(i*t_total_i)*100.0
       efficiency_transitoire[i-1] = t_transitoire_0/(t_transitoire_i*i)*100.0

with open('run_' + version + '.log', 'a') as log:
    s = 'last diff (sequential) = ' + "{:10.4f}".format(last_diff_0) + "\n\n"
    sys.stdout.write(s)
    log.write(s)

    s = "threads:  " +  "".join(["{:10d}".format(u) for u in threads]) + "\n"
    sys.stdout.write(s)
    log.write(s)
    
    s = "speedups: " + "".join(["{:10.3f}".format(u) for u in speedup]) + "\n"
    sys.stdout.write(s)
    log.write(s)
    
    s = "speedups (transitoire): " + "".join(["{:10.3f}".format(u) for u in speedup_transitoire]) + "\n"
    sys.stdout.write(s)
    log.write(s)
    
    s = "last diff:" + "".join(["{:10.4g}".format(u) for u in last]) + "\n"
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


fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(14,4))

ax1.plot(threads, t_total, 'o-')
ax1.plot(threads, t_transitoire, 'o-')
ax1.set_title('Temps CPU (s)')
ax1.xaxis.set_ticks(range(1,args.threadsMax+1))
ax1.set_xlabel('Threads')
ax1.legend([unicode('total'), unicode('transitoire')])
ax1.grid()

ax2.plot(threads, speedup, 'o-')
ax2.plot(threads, speedup_transitoire, 'o-')
ax2.plot(threads, threads, '-')
plt.ylim(0, args.threadsMax)
ax2.legend([unicode('total'), unicode('transitoire'), unicode('ideal')])
ax2.set_xlabel('Threads')
ax2.set_title('Speedup')
ax2.xaxis.set_ticks(range(1,args.threadsMax+1))
ax2.grid()

ax3.plot(threads, efficiency, 'o-')
ax3.plot(threads, efficiency_transitoire, 'o-')
ax3.plot([1, args.threadsMax], [100, 100], '-')
plt.ylim(0, 120)
ax3.legend([unicode('total'), unicode('transitoire'), unicode('ideal')])
ax3.set_xlabel('Threads')
ax3.set_title('Efficiency (%)')
ax3.xaxis.set_ticks(range(1,args.threadsMax+1))
ax3.grid()
fig.tight_layout()
plt.savefig("speedups_OpenMP_CoarseGrain.pdf")

try:
   if display:
      plt.show()
except:
    pass



