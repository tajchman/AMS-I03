#! /usr/bin/env python

import subprocess, os, sys, math, argparse
import matplotlib, numpy
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=12)
args = parser.parse_args()

n = args.n
x = numpy.zeros(n)
m1 = numpy.zeros(n)
m2 = numpy.zeros(n)

k = 1
for i in range(n):
    k *= 2
    x[i] = k
    out = subprocess.check_output(['./ex_1_2.exe', str(k)], text=True)
    p = out.find("lignes   ") + 9
    m1[i] = float(out[p:].split()[0])
    p = out.find("colonnes ") + 9
    m2[i] = float(out[p:].split()[0])
    print (out)

plt.figure()

plt.semilogy(x, m1, "ro-", label="algo. lignes")
plt.semilogy(x, m2, "bo-", label="algo. colonnes")
plt.xlabel('taille matrice')
plt.ylabel('temps CPU')
plt.legend()

plt.savefig('matrices.pdf') 
plt.show()
