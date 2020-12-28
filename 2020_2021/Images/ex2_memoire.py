from figures import *

L = 30
H = 4

class Scene0:
  def __init__(self, L, H):
    self.L = L
    self.H = H

  def plot(self, p):
    self.q0 = Point(1,1)
    self.q1= self.q0
    for i in range(3):
      p.add(Text(self.q1, '$M_{' + str(i) + ',0}$'))
      p.add(Text(self.q1 + (2,0), '$M_{' + str(i) + ',1}$'))
      p.add(Text(self.q1 + (4,0), '\ldots'))
      p.add(Text(self.q1 + (6,0), '$M_{' + str(i) + ',m}$'))
      p.add(Line(self.q1 + (-0.2,-0.3), self.q1 + (-0.2, 0.8), linewidth=0.1))
      p.add(Line(self.q1 + (1.8,-0.3), self.q1 + (1.8, 0.8)))
      p.add(Line(self.q1 + (3.8,-0.3), self.q1 + (3.8, 0.8)))
      p.add(Line(self.q1 + (5.8,-0.3), self.q1 + (5.8, 0.8)))
      self.q1  = self.q1 + (8.3, 0)
    
    p.add(Line(self.q1 + (-0.2,-0.3), self.q1 + (-0.2, 0.8), linewidth=0.1))
    p.add(Text(self.q1, '\ldots'))
    p.add(Rectangle(Point(0.8, 0.7), self.q1 + (2,0.8), linewidth=0.1))

  def plot2(self, p):
    p.add(Rectangle((0,0), self.q1 + (3,2)))
    p.add(Text((0.3, 2.8), r"M\'emoire centrale"))

class Scene1(Scene0):
  def __init__(self, L, H):
    super().__init__(L, H)

  def plot(self, p):
    super().plot(p)

  def plot2(self, p):
    q = self.q0 + (0.7, 0.8)
    for i in range(3):
      if i==0:
        color = 'red'
      elif i==1:
        color = 'green'
      elif i==2:
        color = 'blue'

      for j in range(4):
        p1 = q
        p4 = p1 + (2,0)
        if j == 3:
          p4 = p4 + (0.3,0)
        p2 = (p1*2 + p4)/3 + (0,1)
        p3 = (p1 + p4*2)/3 + (0,1)
        p.add(Bezier(p1, p2, p3, p4, linecolor=color, linewidth=0.15, arrow='->', arrowsize=0.5))
        q = p4
    pass

class Scene2(Scene0):
  def __init__(self, L, H):
    super().__init__(L, H)

  def plot(self, p):
    super().plot(p)

  def plot2(self, p):
    for i in range(3):
      q = self.q0 + Point(i*2, 0) + (0.7, 0.8)
      if i==0:
        color = 'red'
      elif i==1:
        color = 'green'
      elif i==2:
        color = 'blue'
        q = q + Point(2,0)

      for j in range(4):
        p1 = q
        p4 = p1 + (8.3,0)
        if j == 3:
          p4 = p4 + (0.3,0)
        p2 = (p1*2 + p4)/3 + (0,2)
        p3 = (p1 + p4*2)/3 + (0,2)
        p.add(Bezier(p1, p2, p3, p4, linecolor=color, linewidth=0.15, arrow='->', arrowsize=0.5))
        q = p4
    pass

p = pstricks(0,0, 29, 4, font="huge", fontname="ptm")
S = Scene0(L, H)

p.open('ex2_memoire.tex')
S.plot(p)
S.plot2(p)
p.close()

S = Scene1(L, H-1)

p = pstricks(0,0, 29, 3.5, font="huge", fontname="ptm")
p.open('ex2_memoire_1.tex')
S.plot(p)
S.plot2(p)
p.close()

S = Scene2(L, H)

p.open('ex2_memoire_2.tex')
S.plot(p)
S.plot2(p)
p.close()

