from figures import *

L = 14
H = 16

p = pstricks(0,0, L, H, font="huge", fontname="ptm")
p.open('enchainement.tex')

p1 = Point(5, H-0.2)
p2 = p1 - (0, 1.5)
p3 = p2 - (0, 1)
p.add(Line(p1, p2, arrow = '->', arrowsize=0.5, linewidth=0.15, linecolor='red'))
p.add(Line(p1, p3, linewidth=0.15, linecolor='red'))

p4 = p3 - (0, 4)
for i in range(4):
  q1 = p3 + (i-1.5, -0.5)
  q2 = q1 - (0, 2)
  q3 = q2 - (0, 1)
  if i == 0:
    c = 'red'
  else:
    c = 'blue'
  p.add(Lines((p3, q1, q3, p4), linewidth=0.15, linecolor=c))
  p.add(Line(q1, q2, arrow = '->', arrowsize=0.5, linewidth=0.15, linecolor=c))

p5 = p4 - (0, 1.5)
p6 = p5 - (0, 1)
p.add(Line(p4, p5, arrow = '->', arrowsize=0.5, linewidth=0.15, linecolor='red'))
p.add(Line(p4, p6, linewidth=0.15, linecolor='red'))

p7 = p6 - (0, 4)
for i in range(5):
  q1 = p6 + (i-2, -0.5)
  q2 = q1 - (0, 2)
  q3 = q2 - (0, 1)
  if i == 0:
    c = 'red'
  else:
    c = 'blue'
    
  p.add(Lines((p6, q1, q3, p7), linewidth=0.15, linecolor=c))
  p.add(Line(q1, q2, arrow = '->', arrowsize=0.5, linewidth=0.15, linecolor=c))

p8 = p7 - (0, 1.5)
p9 = p8 - (0, 1)
p.add(Line(p7, p8, arrow = '->', arrowsize=0.5, linewidth=0.15, linecolor='red'))
p.add(Line(p7, p9, linewidth=0.15, linecolor='red'))

p.add(Line((0.7, H-0.2), (1.3, H-0.2), linewidth=0.1))
p.add(Line((1, H-0.2), (1, 0.2), linewidth=0.1, arrow='->', arrowsize=0.5))
p.add(Text((1.2,1), "Temps"))
p.add(Text((1.2,0.2), "CPU"))

p.add(Line(p1 + (3,0),p3 + (3,0), arrow='<->', linewidth=0.1, arrowsize=0.3))
p.add(Text(p1*0.5 + p3*0.5 + (3.2,-0.2), 'région séquentielle'))
p.add(Line(p3 + (3,0),p4 + (3,0), arrow='<->', linewidth=0.1, arrowsize=0.3))
p.add(Text(p3*0.5 + p4*0.5 + (3.2,-0.2), 'région parallèle'))
p.add(Line(p4 + (3,0),p6 + (3,0), arrow='<->', linewidth=0.1, arrowsize=0.3))
p.add(Text(p4*0.5 + p6*0.5 + (3.2,-0.2), 'région séquentielle'))
p.add(Line(p6 + (3,0),p7 + (3,0), arrow='<->', linewidth=0.1, arrowsize=0.3))
p.add(Text(p6*0.5 + p7*0.5 + (3.2,-0.2), 'région parallèle'))
p.add(Line(p7 + (3,0),p9 + (3,0), arrow='<->', linewidth=0.1, arrowsize=0.3))
p.add(Text(p7*0.5 + p9*0.5 + (3.2,-0.2), 'région séquentielle'))
p.close()
