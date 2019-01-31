#! /usr/bin/env python

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

with PdfPages('offset_stride.pdf') as pdf:
  x, y = np.loadtxt('offset.txt', delimiter=' ', unpack=True)
  y /= y[0]
  plt.plot(x,y, 'r-')
  plt.ylim(0.0, 1.2)
  plt.xlabel('Offset')
  plt.ylabel('Time Offset/Time Offset 1')
  pdf.savefig()
  plt.close()
  
  x, y = np.loadtxt('stride.txt', delimiter=' ', unpack=True)
  y /= y[0]
  plt.plot(x,y, 'b-')
  plt.ylim(0.0, 22.)
  plt.xlabel('Stride')
  plt.ylabel('Time Stride/Time No Stride')
  pdf.savefig()
  plt.close()


