#! /usr/bin/env python

import os, sys, subprocess, argparse

parser = argparse.ArgumentParser()
parser.add_argument('threadsMax', type=int)
parser.add_argument('-c', '--compilers', default='gnu')
parser.add_argument('-t', '--type', default='Release', 
                    choices=['Release', 'Debug'])
parser.add_argument('rest', nargs=argparse.REMAINDER)
args = parser.parse_args()

base = os.path.join('.', 
                    'install', 
                    args.compilers,
                    args.type)

resultsDir = os.path.join('.', 'results', args.compilers, args.type)
if not os.path.exists(resultsDir):
   os.makedirs(resultsDir)

codeSeq = os.path.join(base, 'PoissonSeq.exe')
codePar = os.path.join(base, 'PoissonOpenMP_FineGrain.exe')

subprocess.call([codeSeq, "path=" + resultsDir] + args.rest)
for i in range(1,args.threadsMax+1):
    subprocess.call([codePar, 'threads=' + str(i), "path=" + resultsDir] + args.rest)

with open('./.run.py', 'w') as f:
    f.write('resultsDir = "' + resultsDir + '"\n')
    f.write('threads = ' + str(args.threadsMax) + '\n')
    
t = []
x = []
speedup = []

def readTemps(n):
    s = os.path.join(resultsDir, 'temps_' + str(n) + '.dat')
        
    with open(s) as f:
        l = f.readline().split()
        return float(l[1])
        
    return 0.0

x0 = readTemps(0)

for i in range(1,args.threadsMax+1):
    u = readTemps(i)
    t.append(i)
    x.append(u)
    speedup.append(x0/u)

print("cpu times:", "".join(["{:9.3f}".format(u) for u in x]))
print("speedups: ", "".join(["{:9.3f}".format(u) for u in speedup]))

import matplotlib.pyplot as plt


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,16))

ax1.plot(t, x, 'o-')
ax1.set_ylabel('Temps CPU (s)')
ax1.xaxis.set_ticks(range(1,args.threadsMax+1))
ax1.grid()

ax2.plot(t, speedup, 'o-')
ax2.plot(t, t, '-')
plt.ylim(0, args.threadsMax)
ax2.legend(['mesuré', 'idéal'])
ax2.set_xlabel('Threads')
ax2.set_ylabel('Speedup')
ax2.xaxis.set_ticks(range(1,args.threadsMax+1))
ax2.grid()

fig.set_size_inches(5, 7)
plt.show()


