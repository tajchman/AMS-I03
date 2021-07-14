#! /usr/bin/env python

import subprocess, os, sys, math
import matplotlib, numpy
import matplotlib.pyplot as plt

n= 1000
m = 8

m1 = numpy.zeros(m)
m2 = numpy.zeros(m)

for i in range(n):

    subprocess.call(['./ex_1_1.exe'])
    with open('results.dat') as f:
        for l in f:
            ll = l.split()
            it = int(ll[0]) - 1
            cy = float(ll[1])
            if i == 0:
                m1[it] = cy
                m2[it] = 0.0
            else:
                m1[it] += cy
                d = (i+1)*cy - m1[it]
                m2[it] += (d*d)/(i*(i+1))

for i in range(m):
    m1[i] /= n
    m2[i] = math.sqrt(m2[i]/n)

plt.figure()
x=numpy.linspace(1,m, num=m)
print(x)
print(m1)
print(m2)
p1 = plt.plot(x, m1, "bo", label="moyenne")
p2 = plt.errorbar(x, m1, yerr=m2, fmt='o', ecolor='r', capsize=4, label="écart-type")
plt.xlabel('Itération')
plt.ylabel('Cycles')
plt.legend()
plt.title('Nombre de cycles pour une iteration')
plt.savefig('cycles.pdf') 
plt.show()
