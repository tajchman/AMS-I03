#! /usr/bin/env python

import os, sys, glob, subprocess, argparse

parser = argparse.ArgumentParser()
parser.add_argument('threads', type=int)
args = parser.parse_args()

t = []
x = []
speedup = []

def readTemps(n):
    if n == 0:
        s = '../PoissonSeq/temps_0.dat'
    else:
        s = 'temps_' + str(n) + '.dat'
        
    with open(s) as f:
        l = f.readline().split()
        return float(l[1]);
        
    return 0.0;

x0 = readTemps(0)

for i in range(1,args.threads+1):
    u = readTemps(i)
    t.append(i)
    x.append(u)
    speedup.append(x0/u)

print(x)
print(speedup)

import matplotlib.pyplot as plt


fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(t, x, 'o-')
ax2.set_xlabel('Threads')
ax1.set_ylabel('Temps CPU')

ax2.plot(t, speedup, 'o-')
ax2.plot(t, t, 'o-')
ax2.set_xlabel('Threads')
ax2.set_ylabel('Speedup')

plt.show()


