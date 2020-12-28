from figures import *

L = 16
H = 12

class Scene0:
   def __init__(self, L, H):
      self.p1 = Point(L/5., H-3.1)
      self.p2 = Point(3.5*L/5., H-3.1)
      pass

   def plot(self, p):
      p.add(Circle(self.p1, 1.0))
      p.add(Circle(self.p2, 1.0))
      p.add(Text(self.p1 + (1.2, 0), 'C\oe ur 1'))
      p.add(Text(self.p2 + (1.2, 0), 'C\oe ur 2'))
      p.add(Rectangle(self.p1 + (-1.3,-5.5), self.p1 + (1.3,-1.5)))
      p.add(Rectangle(self.p2 + (-1.3,-5.5), self.p2 + (1.3,-1.5)))
      p.add(Text(self.p1 + (1.7, -2), 'Mémoire'))
      p.add(Text(self.p1 + (1.7, -2.7), 'cache 1'))
      p.add(Text(self.p2 + (1.7, -2), 'Mémoire'))
      p.add(Text(self.p2 + (1.7, -2.7), 'cache 2'))

      p.add(Rectangle((0,0), (L,3)))
      p.add(Text((0.3, 2.3), r"M\'emoire"))
      p.add(Text((3.5,0.7), '$v$'))
      p.add(Text((3.5,1.9), '$u$'))
      for i in range(5,11):
         j = i*1.1
         p.add(Text((j-1,0.7), '$v_' + str(i-5) + '$'))
         p.add(Text((j-1,1.9), '$u_' + str(i-5) + '$'))
         p.add(Line((j,0.4), (j,1.3)))
         p.add(Line((j,1.6), (j,2.5)))
      p.add(Text((11.1,0.7), '\ldots'))
      p.add(Text((11.1,1.9), '\ldots'))
      p.add(Rectangle((4.4,0.4), (12,1.3)))
      p.add(Rectangle((4.4,1.6), (12,2.5)))

class Scene1(Scene0):
   def __init__(self, L, H):
      super().__init__(L, H)

   def plot(self, p):

      q0 = Point(2.1, 6.7)
      p.add(Text(q0, '$u_0$'))
      q1 = q0 + (1.2, 0)
      p.add(Text(q1, '$u_1$'))
      q2 = q0 + (2.2, 0.55)
      p.add(Rectangle(q0 + (0, -0.3), q2, linecolor='blue', linewidth=0.07))

      q0 = q0 + Point(0, -1)
      p.add(Text(q0, '$u_2$'))
      q1 = q0 + (1.2, 0)
      p.add(Text(q1, '$u_3$'))
      q2 = q0 + (2.2, 0.55)
      p.add(Rectangle(q0 + (0, -0.3), q2, linecolor='blue', linewidth=0.07))

      q0 = q0 + Point(0, -1)
      p.add(Text(q0, '$v_2$'))
      q1 = q0 + (1.2, 0)
      p.add(Text(q1, '$v_3$'))
      q2 = q0 + (2.2, 0.55)
      p.add(Rectangle(q0 + (0, -0.3), q2, linecolor='red', linewidth=0.07))

      q0 = Point(10.1, 6.7)
      p.add(Text(q0, '$u_2$'))
      q1 = q0 + (1.2, 0)
      p.add(Text(q1, '$u_3$'))
      q2 = q0 + (2.2, 0.55)
      p.add(Rectangle(q0 + (0, -0.3), q2, linecolor='blue', linewidth=0.07))

      q0 = q0 + Point(0, -1)
      p.add(Text(q0, '$u_4$'))
      q1 = q0 + (1.2, 0)
      p.add(Text(q1, '$u_5$'))
      q2 = q0 + (2.2, 0.55)
      p.add(Rectangle(q0 + (0, -0.3), q2, linecolor='blue', linewidth=0.07))

      q0 = q0 + Point(0, -1)
      p.add(Text(q0, '$v_2$'))
      q1 = q0 + (1.2, 0)
      p.add(Text(q1, '$v_3$'))
      q2 = q0 + (2.2, 0.55)
      p.add(Rectangle(q0 + (0, -0.3), q2, linecolor='red', linewidth=0.07))

      super().plot(p)
      q1 = Point(6.65, 0.45)
      q2 = q1 + (2.1, 0.8)
      p.add(Rectangle(q1, q2, linecolor='red', linewidth=0.07))
      q1 = Point(4.45, 1.65)
      q2 = Point(6.55, 2.45)
      p.add(Rectangle(q1, q2, linecolor='blue', linewidth=0.07))
      q1 = q1 + (2.2, 0)
      q2 = q2 + (2.2, 0)
      p.add(Rectangle(q1, q2, linecolor='blue', linewidth=0.07))
      q1 = q1 + (2.2, 0)
      q2 = q2 + (2.2, 0)
      p.add(Rectangle(q1, q2, linecolor='blue', linewidth=0.07))

      q1 = Point(5.5, 2.45)
      q4 = Point(3.8, 7.0)
      q2 = (q1*2 + q4)/3 + (1,0)
      q3 = (q1 + q4*2)/3 + (1,0)
      p.add(Bezier(q1, q2, q3, q4 + (0.5, -0.1), \
                       linecolor='blue', arrow='->', linewidth=0.1, arrowsize=0.4))
      q1 = Point(7.7, 2.45)
      q4 = Point(3.8, 6.0)
      q2 = (q1*2 + q4)/3 + (1,0)
      q3 = (q1 + q4*2)/3 + (1,0)
      p.add(Bezier(q1, q2, q3, q4 + (0.5, -0.1), \
                       linecolor='blue', arrow='->', linewidth=0.1, arrowsize=0.4))
      q1 = Point(6.65, 0.8)
      q4 = Point(2.6, 4.5)
      q2 = (q1*2 + q4)/3 + (-1,-1)
      q3 = (q1 + q4*2)/3 + (-1,-2)
      p.add(Bezier(q1, q2, q3, q4 + (0.5, -0.1), \
                       linecolor='red', arrow='->', linewidth=0.1, arrowsize=0.4))

      q1 = Point(9.9, 2.45)
      q4 = Point(9.6, 5.9)
      q2 = (q1*2 + q4)/3 + (-0.5,0)
      q3 = (q1 + q4*2)/3 + (-0.5,0)
      p.add(Bezier(q1, q2, q3, q4 + (0.5, -0.1), \
                       linecolor='blue', arrow='->', linewidth=0.1, arrowsize=0.4))
      q1 = Point(7.7, 2.45)
      q4 = Point(9.6, 7.0)
      q2 = (q1*2 + q4)/3 + (0,0)
      q3 = (q1 + q4*2)/3 + (-1,0)
      p.add(Bezier(q1, q2, q3, q4 + (0.5, -0.1), \
                       linecolor='blue', arrow='->', linewidth=0.1, arrowsize=0.4))
      q1 = Point(8.7, 0.8)
      q4 = Point(10.65, 4.5)
      q2 = (q1*2 + q4)/3 + (4,-1)
      q3 = (q1 + q4*2)/3 + (1,-2)
      p.add(Bezier(q1, q2, q3, q4 + (0.5, -0.1), \
                       linecolor='red', arrow='->', linewidth=0.1, arrowsize=0.4))

class Scene2(Scene0):
   def __init__(self, L, H):
        super().__init__(L, H)

   def plot(self, p):
      p.add(Text(self.p1 + (-3, 1.7), r'\LARGE $w_2 = (u_1 + 2u_2 + u_3)/4$'))
      p.add(Text(self.p2 + (-3, 1.7), r'\LARGE $w_3 = (u_2 + 2u_3 + u_4)/4$'))
      p.add(Arc(self.p1, 0.6, -30, 330, linewidth=0.07, arrow="->", arrowinset=0, arrowsize="3pt 3"))
      p.add(Arc(self.p2, 0.6, -30, 330, linewidth=0.07, arrow="->", arrowinset=0, arrowsize="3pt 3"))

      q0 = Point(2.1, 6.7)
      p.add(Text(q0, '$u_0$'))
      q1 = q0 + (1.2, 0)
      p.add(Text(q1, '$u_1$'))
      q2 = q0 + (2.2, 0.55)
      p.add(Rectangle(q1 + (0, -0.3), q2, linecolor='blue', linewidth=0.07))

      q0 = q0 + Point(0, -1)
      p.add(Text(q0, '$u_2$'))
      q1 = q0 + (1.2, 0)
      p.add(Text(q1, '$u_3$'))
      q2 = q0 + (2.2, 0.55)
      p.add(Rectangle(q0 + (0, -0.3), q1 + (-0.2, 0.55), linecolor='blue', linewidth=0.07))
      p.add(Rectangle(q1 + (0, -0.3), q2, linecolor='blue', linewidth=0.07))

      q0 = q0 + Point(0, -1)
      p.add(Text(q0, '$w_2$'))
      q1 = q0 + (1.2, 0)
      p.add(Text(q1, '$v_3$'))
      q2 = q0 + (2.2, 0.55)
      p.add(Rectangle(q0 + (0, -0.3), q1 + (-0.2, 0.55), linecolor='red', linewidth=0.07))

      q0 = Point(10.1, 6.7)
      p.add(Text(q0, '$u_2$'))
      q1 = q0 + (1.2, 0)
      p.add(Text(q1, '$u_3$'))
      q2 = q0 + (2.2, 0.55)
      p.add(Rectangle(q0 + (0, -0.3), q1 + (-0.2, 0.55), linecolor='blue', linewidth=0.07))
      p.add(Rectangle(q1 + (0, -0.3), q2, linecolor='blue', linewidth=0.07))

      q0 = q0 + Point(0, -1)
      p.add(Text(q0, '$u_4$'))
      q1 = q0 + (1.2, 0)
      p.add(Text(q1, '$u_5$'))
      q2 = q0 + (2.2, 0.55)
      p.add(Rectangle(q0 + (0, -0.3), q1 + (-0.2, 0.55), linecolor='blue', linewidth=0.07))

      q0 = q0 + Point(0, -1)
      p.add(Text(q0, '$v_2$'))
      q1 = q0 + (1.2, 0)
      p.add(Text(q1, "$w_3$"))
      q2 = q0 + (2.2, 0.55)
      p.add(Rectangle(q1 + (0, -0.3), q2, linecolor='red', linewidth=0.07))

      super().plot(p)

      q1 = Point(3.8, 7.2)
      q4 = Point(4.0, 8.3)
      q2 = (q1*2 + q4)/3 + (0.1,0)
      q3 = (q1 + q4*2)/3 + (0.1,0)
      p.add(Bezier(q1, q2, q3, q4, \
                       linecolor='blue', arrow='->', linewidth=0.1, arrowsize=0.4))
      q1 = Point(4.3, 5.8)
      q4 = Point(4.2, 9.0)
      q2 = (q1*2 + q4)/3 + (0.5,0)
      q3 = (q1 + q4*2)/3 + (1,0)
      p.add(Bezier(q1, q2, q3, q4, \
                       linecolor='blue', arrow='->', linewidth=0.1, arrowsize=0.4))
      q1 = Point(2.1, 5.8)
      q4 = Point(2.45, 8.25)
      q2 = (q1*2 + q4)/3 + (-1.0,0)
      q3 = (q1 + q4*2)/3 + (-1,0)
      p.add(Bezier(q1, q2, q3, q4, \
                       linecolor='blue', arrow='->', linewidth=0.1, arrowsize=0.4))
      q1 = Point(2.1, 4.8)
      q4 = Point(2.2, 8.9)
      q2 = (q1*2 + q4)/3 + (-1.5,0)
      q3 = (q1 + q4*2)/3 + (-1.5,0)
      p.add(Bezier(q1, q2, q3, q4, \
                       linecolor='red', arrow='<-', linewidth=0.1, arrowsize=0.4))

      q1 = Point(11.8, 7.2)
      q4 = Point(12.0, 8.3)
      q2 = (q1*2 + q4)/3 + (0.5,0)
      q3 = (q1 + q4*2)/3 + (0.2,0)
      p.add(Bezier(q1, q2, q3, q4, \
                       linecolor='blue', arrow='->', linewidth=0.1, arrowsize=0.4))
      q1 = Point(10.6, 7.2)
      q4 = Point(10.35, 8.4)
      q2 = (q1*2 + q4)/3 + (-0.2, 0)
      q3 = (q1 + q4*2)/3 + (-0.2,-0.4)
      p.add(Bezier(q1, q2, q3, q4, showpoints='false',\
                       linecolor='blue', arrow='->', linewidth=0.1, arrowsize=0.4))
      q1 = Point(10.1, 5.8)
      q4 = Point(10.2, 8.85)
      q2 = (q1*2 + q4)/3 + (-1.0,0)
      q3 = (q1 + q4*2)/3 + (-1,0)
      p.add(Bezier(q1, q2, q3, q4, \
                       linecolor='blue', arrow='->', linewidth=0.1, arrowsize=0.4))
      q1 = Point(12.1, 8.6)
      q4 = Point(12.25, 4.8)
      q2 = (q1*2 + q4)/3 + (1.,0)
      q3 = (q1 + q4*2)/3 + (1.,0)
      p.add(Bezier(q1, q2, q3, q4, \
                       linecolor='red', arrow='->', linewidth=0.1, arrowsize=0.4))



class Scene3(Scene0):
    def __init__(self, L, H):
        super().__init__(L, H)

    def plot(self, p):
      q0 = Point(2.1, 6.7)
      p.add(Text(q0, '$u_0$'))
      q1 = q0 + (1.2, 0)
      p.add(Text(q1, '$u_1$'))
      q2 = q0 + (2.2, 0.55)

      q0 = q0 + Point(0, -1)
      p.add(Text(q0, '$u_2$'))
      q1 = q0 + (1.2, 0)
      p.add(Text(q1, '$u_3$'))
      q2 = q0 + (2.2, 0.55)

      q0 = q0 + Point(0, -1)
      s0 = q0
      p.add(Text(q0, '$w_2$'))
      q1 = q0 + (1.2, 0)
      p.add(Text(q1, '$v_3$'))
      q2 = q0 + (2.2, 0.55)
      p.add(Rectangle(q0 + (0, -0.3), q2, linecolor='red', linewidth=0.07))

      q0 = Point(10.1, 6.7)
      p.add(Text(q0, '$u_2$'))
      q1 = q0 + (1.2, 0)
      p.add(Text(q1, '$u_3$'))
      q2 = q0 + (2.2, 0.55)

      q0 = q0 + Point(0, -1)
      p.add(Text(q0, '$u_4$'))
      q1 = q0 + (1.2, 0)
      p.add(Text(q1, '$u_5$'))
      q2 = q0 + (2.2, 0.55)

      q0 = q0 + Point(0, -1)
      s1 = q0
      p.add(Text(q0, '$v_2$'))
      q1 = q0 + (1.2, 0)
      p.add(Text(q1, '$w_3$'))
      q2 = q0 + (2.2, 0.55)
      p.add(Rectangle(q0 + (0, -0.3), q2, linecolor='red', linewidth=0.07))

      super().plot(p)

      p.add(Text((6.7,0.6), '??'))
      p.add(Text((7.8,0.6), '??'))
      r0 = Point(6.65, 0.45)
      r1 = r0 + (2.1, 0.8)
      p.add(Rectangle(r0, r1, linecolor='red', linewidth=0.07))

      r1 = s0 + (2.15, 0.1)
      r4 = Point(7.7, 1.3)
      r2 = r1 + (3.,0)
      r3 = r4 + (0.,1)
      p.add(Bezier(r1, r2, r3, r4, \
                       linecolor='red', arrow='->', linewidth=0.1, arrowsize=0.4))
      r1 = s1 + (0, 0.1)
      r4 = Point(7.7, 1.3)
      r2 = r1 + (-2.,0)
      r3 = r4 + (0.,1)
      p.add(Bezier(r1, r2, r3, r4, \
                       linecolor='red', arrow='->', linewidth=0.1, arrowsize=0.4))


class Scene4(Scene0):
    def plot(self, p):
      q0 = Point(2.1, 6.7)
      p.add(Text(q0, '$u_0$'))
      q1 = q0 + (1.2, 0)
      p.add(Text(q1, '$u_1$'))
      q2 = q0 + (2.2, 0.55)

      q0 = q0 + Point(0, -1)
      p.add(Text(q0, '$u_2$'))
      q1 = q0 + (1.2, 0)
      p.add(Text(q1, '$u_3$'))
      q2 = q0 + (2.2, 0.55)

      q0 = q0 + Point(0, -1)
      s0 = q0
      p.add(Text(q0, '$w_2$'))
      q1 = q0 + (1.2, 0)
      p.add(Text(q1, '$v_3$'))
      q2 = q0 + (2.2, 0.55)
      p.add(Rectangle(q0 + (0, -0.3), q1 + (0, 0.55), linecolor='red', linewidth=0.07))

      q0 = Point(10.1, 6.7)
      p.add(Text(q0, '$u_2$'))
      q1 = q0 + (1.2, 0)
      p.add(Text(q1, '$u_3$'))
      q2 = q0 + (2.2, 0.55)

      q0 = q0 + Point(0, -1)
      p.add(Text(q0, '$u_4$'))
      q1 = q0 + (1.2, 0)
      p.add(Text(q1, '$u_5$'))
      q2 = q0 + (2.2, 0.55)

      q0 = q0 + Point(0, -1)
      p.add(Text(q0, '$v_2$'))
      q1 = q0 + (1.2, 0)
      s1 = q1
      p.add(Text(q1, '$w_3$'))
      q2 = q0 + (2.2, 0.55)
      p.add(Rectangle(q1 + (0, -0.3), q2+ (0.1, 0), linecolor='red', linewidth=0.07))

      super().plot(p)

      r1 = s0 + (1.2, -0.25)
      r4 = s1 + (-1.2, -0.25)
      r2 = (r1*2+r4)/3 + (0.,-1)
      r3 = (r1+r4*2)/3 + (0.,-1)
      p.add(Bezier(r1, r2, r3, r4, \
                       linecolor='red', arrow='->', linewidth=0.1, arrowsize=0.4))
      r1 = s1 + (0, 0.5)
      r4 = s0 + (2.1, 0.4)
      r2 = (r1*2+r4)/3 + (0.,1)
      r3 = (r1+r4*2)/3 + (0.,1)
      p.add(Bezier(r1, r2, r3, r4, \
                       linecolor='red', arrow='->', linewidth=0.1, arrowsize=0.4))


p = pstricks(0,0,L, H, font="huge", fontname="ptm")
S = Scene0(L, H)

p.open('multithread0.tex')
S.plot(p)
p.close()

S = Scene1(L, H)

p.open('multithread1.tex')
S.plot(p)
p.close()

S = Scene2(L, H)

p.open('multithread2.tex')
S.plot(p)
p.close()

S = Scene3(L, H)

p.open('multithread3.tex')
S.plot(p)
p.close()

S = Scene4(L, H)

p.open('multithread4.tex')
S.plot(p)
p.close()
