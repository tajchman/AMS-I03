 
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Creating vectors X and Y 
n = 250
x0 = np.linspace(0, 5, n*4) 
y0 = x0

x1 = np.linspace(0, 5, n*4)
y1 = 0.3*x1*np.log(x1+1)

x2 = np.linspace(0,5, n*4)
y2 = (x2+0.5)*(x2+0.5)-0.25
imax=4*n
for i in range(4*n):
    if y2[i] > 4.6:
        imax = i
        break
x2 = x2[:imax]
y2 = y2[:imax]

 
fs=28
fig = plt.figure(figsize = (10, 10))  
plt.plot(x0, y0, linestyle='dashed', linewidth=3, color='blue') 
plt.plot(x1, y1, linewidth=3, color='green') 
plt.plot(x2, y2, linewidth=3, color='red') 
plt.text(x1[-1]-1.3, y1[-1]+0.1, 'FFT: $O(N\;\log(N))$', fontsize=fs, color='green')
plt.text(x2[-1]-1.1, y2[-1]+0.1, '$N$-corps: $O(N^2)$', fontsize=fs, color='red')

plt.text(1.2, 3.5, "adapté", fontsize=fs, color='red', bbox=dict(facecolor='white', edgecolor='black'))
plt.text(2.8, 1.5, "difficile", fontsize=fs, color='green', bbox=dict(facecolor='white', edgecolor='black'))

plt.arrow(0,0,5,0, width=0.05)
plt.text(1,-0.3, "Nombre d'accès mémoire", fontsize=fs)
plt.arrow(0,0,0,5, width=0.05)
plt.text(-0.3,1, "Nombre de calculs", fontsize=fs, rotation=90)
plt.axis('off')
plt.savefig("GPU_algos.pdf")
