from figures import *

L = 24
H = 8

p = pstricks(0,0,L, H, font="huge", fontname="ptm")

def noeud(p0, proc, nth):
  p.add(Text(p0, "processus $P" + str(proc) + "$"))
  p.add(Rectangle(p0 + (-0.5, 1.0), p0 + (2*nth+0.5, -2.5), linewidth=0.1))

  for i in range(nth):
    q = p0 + (1+ 2*i, -1.3)
    p.add(Text(q, "$T_" + str(i) + "$", justify="c"))
    p.add(Circle(q, 0.8, linewidth=0.08))
    p.add(Line(q + (0, -0.8), q + (0, -2.8), arrow="<->", linewidth=0.2, linecolor="red"))
  
  p1 = p0 + (0, -5)
  p.add(Text(p1, "mémoire $M_" + str(proc) + "$"))
  p.add(Rectangle(p1 + (-0.5, 1.0), p1 + (2*nth+0.5, -1), linewidth=0.1))


p.open('modeleHybride.tex')

x1 = 1
for i in range(3):
  x0 = x1
  p0 = Point(x0,6.5)
  if i==0:
    nth = 3
  if i==1:
    nth = 4
  if i==2:
    nth = 2
  
  noeud(p0, i, nth)
  x1 = x0 + (nth+1)*2
  
p.close()

