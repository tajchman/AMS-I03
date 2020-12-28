from figures import *

L = 30
H = 23

p = pstricks(0,0, L, H, font="huge", fontname="ptm")
p.open('enchainement_coarse.tex')

p1 = Point(5, H-7)

p.add(Text(p1 + (0,6), 'Parallélisme multithreads', justify="c"))
p.add(Text(p1 + (0,5.2), 'grain fin (fine grain)', justify="c"))
p.add(Text(p1 + (0,4), '{\LARGE (threads $T_i, i>0$ créés au début}', justify="c"))
p.add(Text(p1 + (0,3.3), '{\LARGE de chaque région parallèle et détruits}', justify="c"))
p.add(Text(p1 + (0,2.6), '{\LARGE à la fin de chaque région parallèle)}', justify="c"))

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

p.add(Line(p1 + (3.7,0),p3 + (3.7,0), arrow='<->', linewidth=0.1, arrowsize=0.3))
p.add(Text(p1*0.5 + p3*0.5 + (4,-0.2), 'région séquentielle'))
p.add(Line(p3 + (3.7,0),p4 + (3.7,0), arrow='<->', linewidth=0.1, arrowsize=0.3))
p.add(Text(p3*0.5 + p4*0.5 + (4,-0.2), 'région parallèle'))
p.add(Line(p4 + (3.7,0),p6 + (3.7,0), arrow='<->', linewidth=0.1, arrowsize=0.3))
p.add(Text(p4*0.5 + p6*0.5 + (4,-0.2), 'région séquentielle'))
p.add(Line(p6 + (3.7,0),p7 + (3.7,0), arrow='<->', linewidth=0.1, arrowsize=0.3))
p.add(Text(p6*0.5 + p7*0.5 + (4,-0.2), 'région parallèle'))
p.add(Line(p7 + (3.7,0),p9 + (3.7,0), arrow='<->', linewidth=0.1, arrowsize=0.3))
p.add(Text(p7*0.5 + p9*0.5 + (4,-0.2), 'région séquentielle'))

p.add(TextFront(p3 + (-1.8, 0.1), "{\LARGE Threads 1-3 créés}"))
p.add(TextFront(p4 + (-1.8, -0.8), "{\LARGE Threads 1-3 détruits}"))
p.add(Line(p3 - (1.3,0), p3 + (1.3,0), linewidth = 0.12))
p.add(Line(p4 - (1.3,0), p4 + (1.3,0), linewidth = 0.12))

p.add(TextFront(p6 + (-1.8, 0.1), "{\LARGE Threads 1-4 créés}"))
p.add(TextFront(p7 + (-1.8, -0.8), "{\LARGE Threads 1-4 détruits}"))
p.add(Line(p6 - (1.3,0), p6 + (2.3,0), linewidth = 0.12))
p.add(Line(p7 - (1.3,0), p7 + (2.3,0), linewidth = 0.12))


p1 = Point(20, H-7)
p.add(Text(p1 + (0,6), 'Parallélisme multithreads', justify="c"))
p.add(Text(p1 + (0,5.2), 'grain grossier (coarse grain)', justify="c"))

p.add(Text(p1 + (0,4), "{\LARGE (threads $T_i, i>0$ créés au début de l'execution}", justify="c"))
p.add(Text(p1 + (0,3.3), '{\LARGE activés au début de chaque région parallèle}', justify="c"))
p.add(Text(p1 + (0,2.6), '{\LARGE et mis en pause à la fin de chaque région parallèle)}', justify="c"))

p2 = p1 - (0, 1.5)
p3 = p2 - (0, 1)

p4 = p3 - (0, 3.5)
for i in range(1, 4):
  q1 = p3 + (i-2, 0)
  q3 = q1 - (0, 3.5)
  q2 = q1*0.5 + q3*0.5
  p.add(Line(q1, q2, arrow = '->', arrowsize=0.5, linewidth=0.15, linecolor="blue"))
  p.add(Line(q1, q3, linewidth=0.15, linecolor="blue"))

p5 = p4 - (0, 1.5)
p6 = p5 - (0, 1)

p7 = p6 - (0, 3.5)
for i in range(1, 5):
  q1 = p6 + (i-2, 0)
  q3 = q1 - (0, 3.5)
  q2 = q1*0.5 + q3*0.5
  p.add(Line(q1, q2, arrow = '->', arrowsize=0.5, linewidth=0.15, linecolor="blue"))
  p.add(Line(q1, q3, linewidth=0.15, linecolor="blue"))

p8 = p7 - (0, 1.5)
p9 = p8 - (0, 1)

for i in range(1, 5):
  p.add(Line(p1 + (i-2,0), p9 + (i-2,0), 
           linewidth=0.07, linecolor='blue', linestyle="dashed", dash="5pt"))

p19 = p1 *0.5 + p9*0.5 
p.add(Line(p1 - (2,0),p19 - (2,0), arrow = '->', arrowsize=0.5, linewidth=0.15, linecolor='red'))
p.add(Line(p1 - (2,0), p9 - (2,0), linewidth=0.15, linecolor='red'))

p.add(Line(p1 + (3.7,0),p9 + (3.7,0), arrow='<->', linewidth=0.1, arrowsize=0.3))
p.add(Text(p1*0.5 + p9*0.5 + (4.2,-0.2), 'région parallèle'))

p.add(TextFront(p3 + (-1.75, 0.1), "{\LARGE Threads 1-3 activés}"))
p.add(TextFront(p4 + (-1.75, -0.8), "{\LARGE Threads 1-3 en pause}"))
p.add(Line(p3 - (1.3,0), p3 + (1.3,0), linewidth = 0.12))
p.add(Line(p4 - (1.3,0), p4 + (1.3,0), linewidth = 0.12))

p.add(TextFront(p6 + (-1.75, 0.1), "{\LARGE Threads 1-4 activés}"))
p.add(TextFront(p7 + (-1.75, -0.8), "{\LARGE Threads 1-4 en pause}"))
p.add(Line(p6 - (1.3,0), p6 + (2.3,0), linewidth = 0.12))
p.add(Line(p7 - (1.3,0), p7 + (2.3,0), linewidth = 0.12))

p.add(Line((0.7, H-7), (1.3, H-7), linewidth=0.1))
p.add(Line((1, H-7), (1, 0.2), linewidth=0.1, arrow='->', arrowsize=0.5))
p.add(Text((1.2,1), "Temps"))
p.add(Text((1.2,0.2), "CPU"))

q = Point(L/2.-2, H-5.5)

p.add(Text(q, r"\textcolor{red}{\bf Thread 0 actif pendant toute l'exécution}", justify="c"))

p.close()
