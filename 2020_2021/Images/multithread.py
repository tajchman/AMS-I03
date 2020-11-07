from figures import *

L = 16
H = 10

class Scene0:
   def __init__(self, L, H):
      self.p1 = Point(L/4., H-2)
      self.p2 = Point(3*L/4., H-2)
      pass

   def plot(self, p):
      p.add(Circle(self.p1, 1.0))
      p.add(Circle(self.p2, 1.0))
      p.add(Text(self.p1 + (1.2, 0), 'C\oe ur 1'))
      p.add(Text(self.p2 + (1.2, 0), 'C\oe ur 2'))
      p.add(Rectangle((0,0), (L,3)))
      p.add(Text((0.3, 2.3), r"M\'emoire"))
      p.add(Text((5,0.7), '$v$'))
      p.add(Text((5,1.9), '$u$'))
      for i in range(7,12):
         p.add(Text((i-0.9,0.7), '$v_' + str(i-7) + '$'))
         p.add(Text((i-0.9,1.9), '$u_' + str(i-7) + '$'))
         p.add(Line((i,0.5), (i,1.3)))
         p.add(Line((i,1.7), (i,2.5)))
      p.add(Rectangle((6,0.5), (12,1.3)))
      p.add(Rectangle((6,1.7), (12,2.5)))

class Scene1(Scene0):
    def __init__(self, L, H):
        super.__init__(L, H)
        
    def plot(self, p):
        super.plot(p)
        q1 = (6.2,2.5)
        q4 = self.p1 + (-0.5, -1)
        p.add(Bezier(q1, q1 + (-2, 2), q4 + (-1, -2), q4, showpoints='true'))
        
p = pstricks(0,0,L, H, font="huge", fontname="ptm")
S = Scene0(L, H)

p.open('multithread0.tex')
S.plot(p)
p.close()

p.open('multithread1.tex')
S.plot(p)
p.close()

