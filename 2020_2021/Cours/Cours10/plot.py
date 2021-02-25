#! /usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    unicode('')
except NameError:
    unicode = str

import os, sys, argparse
from subprocess import *

u = []
def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def readSpeedups(dirname, filename):
    ll = splitall(dirname)
    print(ll)
    title = ''
    for l in ll:
        k = l.split('_')
        if len(k) < 2:
            continue
        if k[-1] == 'default':
            continue
        title += k[-1] + ','
    if title[-1] == ',':
        title = title[:-1]
    with open(os.path.join(dirname, filename)) as f:
        t = []
        c = []
        s = []
        for line in f:
           l = line.split()
           t.append(int(l[0]))
           c.append(float(l[1]))
           s.append(float(l[2]))
    u.append([title, t, c, s])
    pass

for root, dirs, files in os.walk('.', topdown=False):
   for name in files:
      if name == "speedups.dat":
          readSpeedups(root, name)
print(u)
import matplotlib
display = False

import matplotlib.pyplot as plt


fig, ax = plt.subplots()

ideal = None
tmax=0
legends=[]
for v in u:        
    title = v[0]
    print(title)
    t = v[1]
    c = v[2]
    s = v[3]
    if not ideal:
        ax.plot(t, t, '-')
        tmax = max(t)
        plt.ylim(0, tmax)
        legends.append('ideal')
        ideal = True
    ax.plot(t, s, 'o-')
    legends.append(title)

ax.legend(legends)
ax.set_xlabel('Threads')
ax.set_ylabel('Speedup')
ax.xaxis.set_ticks(range(1,tmax+1))
ax.grid()

fig.set_size_inches(10, 7)
plt.savefig("speedups_OpenMP.pdf")

try:
    if display:
        plt.show()
    else:
        plt.ioff()
except:
    pass



