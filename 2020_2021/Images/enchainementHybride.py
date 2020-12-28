from figures import *

L = 20
H = 18

p = pstricks(0,0, L, H, font="Huge", fontname="ptm")
p.open('enchainementHybride1.tex')

x1 = 7
x2 = 14

p1 = Point(x1, H-2.2)
p2 = p1 - (0, 1)
p3 = p2 - (0, 0.5)
p.add(Line(p1, p2, arrow = '->', arrowsize=0.5, linewidth=0.15, linecolor='red'))
p.add(Line(p1, p3, linewidth=0.15, linecolor='red'))

p4 = p3 - (0, 4)
for i in range(3):
  q1 = p3 + (i-1.0, -0.5)
  q2 = q1 - (0, 2)
  q3 = q2 - (0, 1)
  if i == 0:
    c = 'red'
  else:
    c = 'blue'
  p.add(Lines((p3, q1, q3, p4), linewidth=0.15, linecolor=c))
  p.add(Line(q1, q2, arrow = '->', arrowsize=0.5, linewidth=0.15, linecolor=c))

p5 = p4 - (0, 3)
p6 = p5 - (0, 1.7)
p.add(Line(p4, p5, arrow = '->', arrowsize=0.5, linewidth=0.15, linecolor='red'))
p.add(Line(p4, p6, linewidth=0.15, linecolor='red'))

p7 = p6 - (0, 3)
for i in range(3):
  q1 = p6 + (i-1.0, -0.5)
  q2 = q1 - (0, 1.3)
  q3 = q2 - (0, 0.7)
  if i == 0:
    c = 'red'
  else:
    c = 'blue'
    
  p.add(Lines((p6, q1, q3, p7), linewidth=0.15, linecolor=c))
  p.add(Line(q1, q2, arrow = '->', arrowsize=0.5, linewidth=0.15, linecolor=c))

p8 = p7 - (0, 1.5)
p9 = p8 - (0, 0.8)
p.add(Line(p7, p8, arrow = '->', arrowsize=0.5, linewidth=0.15, linecolor='red'))
p.add(Line(p7, p9, linewidth=0.15, linecolor='red'))

r1 = Point(x2, H-2.2)
r2 = r1 - (0, 1.5)
r3 = r2 - (0, 1)
p.add(Line(r1, r2, arrow = '->', arrowsize=0.5, linewidth=0.15, linecolor='red'))
p.add(Line(r1, r3, linewidth=0.15, linecolor='red'))

r4 = r3 - (0, 4)
for i in range(4):
  q1 = r3 + (i-1.5, -0.5)
  q2 = q1 - (0, 2)
  q3 = q2 - (0, 1)
  if i == 0:
    c = 'red'
  else:
    c = 'blue'
  p.add(Lines((r3, q1, q3, r4), linewidth=0.15, linecolor=c))
  p.add(Line(q1, q2, arrow = '->', arrowsize=0.5, linewidth=0.15, linecolor=c))

r5 = r4 - (0, 1.5)
r6 = r5 - (0, 1)
p.add(Line(r4, r5, arrow = '->', arrowsize=0.5, linewidth=0.15, linecolor='red'))
p.add(Line(r4, r6, linewidth=0.15, linecolor='red'))

r7 = r6 - (0, 4)
for i in range(4):
  q1 = r6 + (i-1.5, -0.5)
  q2 = q1 - (0, 2)
  q3 = q2 - (0, 1)
  if i == 0:
    c = 'red'
  else:
    c = 'blue'
    
  p.add(Lines((r6, q1, q3, r7), linewidth=0.15, linecolor=c))
  p.add(Line(q1, q2, arrow = '->', arrowsize=0.5, linewidth=0.15, linecolor=c))

r8 = r7 - (0, 1.5)
r9 = r8 - (0, 1)
p.add(Line(r7, r8, arrow = '->', arrowsize=0.5, linewidth=0.15, linecolor='red'))
p.add(Line(r7, r9, linewidth=0.15, linecolor='red'))

p.add(Line((0.7, H-2.2), (1.3, H-2.2), linewidth=0.1))
p.add(Line((1, H-2.2), (1, 0.2), linewidth=0.1, arrow='->', arrowsize=0.5))
p.add(Text((1.4,1), "Temps"))
p.add(Text((1.4,0.2), "CPU"))

p.add(Text((x1, H-1.0), "Processus $P_0$", justify='c'))
p.add(Text((x2, H-1.0), "Processus $P_1$", justify='c'))

p.close()
