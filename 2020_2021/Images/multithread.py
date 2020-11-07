from figures import *

L = 16
H = 10

class Scene0:
   def __init__(self, L, H):
      self.p1 = Point(L/5., H-2)
      self.p2 = Point(3.5*L/5., H-2)
      pass

   def plot(self, p):
      p.add(Circle(self.p1, 1.0))
      p.add(Circle(self.p2, 1.0))
      p.add(Text(self.p1 + (1.2, 0), 'C\oe ur 1'))
      p.add(Text(self.p2 + (1.2, 0), 'C\oe ur 2'))
      p.add(Rectangle(self.p1 + (-1.5,-3.5), self.p1 + (1.5,-1.5)))
      p.add(Rectangle(self.p2 + (-1.5,-3.5), self.p2 + (1.5,-1.5)))
      p.add(Text(self.p1 + (1.7, -2), 'Mémoire'))
      p.add(Text(self.p1 + (1.7, -2.7), 'cache 1'))
      p.add(Text(self.p2 + (1.7, -2), 'Mémoire'))
      p.add(Text(self.p2 + (1.7, -2.7), 'cache 2'))

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
        super().__init__(L, H)
        
    def plot(self, p):
        for i in range(1,4):
           q1 = Point(6.5 + i, 2.5)
           q2 = q1 + (-1, 2)
           q4 = self.p1 + (-2.5 + i, -2.1)
           q3 = q4 + (0, -2)
           p.add(Text(q4, '$u_' + str(i) + '$'))
           p.add(Bezier(q1, q2, q3, q4 + (0.5, -0.1), \
                        linecolor='red', arrow='->', linewidth=0.1, arrowsize=0.4))
        for i in range(2,5):
           q1 = Point(6.5 + i, 2.5)
           q2 = q1 + (0, 2)
           q4 = self.p2 + (-3.5 + i, -2.1)
           q3 = q4 + (0, -2)
           p.add(Text(q4, '$u_' + str(i) + '$'))
           p.add(Bezier(q1, q2, q3, q4 + (0.5, -0.1), \
                        linecolor='blue', arrow='->', linewidth=0.1, arrowsize=0.4))
        
        super().plot(p)
 
class Scene2(Scene0):
    def __init__(self, L, H):
        super().__init__(L, H)
        
    def plot(self, p):
        q4 = self.p1
        for i in range(1,4):
           q1 = q4 + (-2.5 + i, -2.1)
           q2 = q4 + (0, -0.5)
           q3 = q4 + (0, -1.3)
           p.add(Text(q1, '$u_' + str(i) + '$'))
           p.add(Bezier(q1 + (0.5, 0.5), q2, q3, q4 , \
                        linecolor='red', arrow='->', linewidth=0.1, arrowsize=0.4, showpoints='false'))

        q4 = self.p2
        for i in range(2,5):
           q1 = q4 + (-3.5 + i, -2.1)
           q2 = q1 + (0.4, 0.5)
           q2 = q4 + (0, -0.5)
           q3 = q4 + (0, -1.3)
           p.add(Text(q1, '$u_' + str(i) + '$'))
           p.add(Bezier(q1 + (0.5, 0.5), q2, q3, q4 , \
                        linecolor='blue', arrow='->', linewidth=0.1, arrowsize=0.4, showpoints='false'))

        super().plot(p)
       
 
class Scene3(Scene0):
    def __init__(self, L, H):
        super().__init__(L, H)
        
    def plot(self, p):
        p.add(Text(self.p1 + (-3, 1.3), r'\LARGE $v_2 = (u_1 + 2u_2 + u_3)/4$'))
        p.add(Text(self.p2 + (-3, 1.3), r'\LARGE $v_3 = (u_2 + 2u_3 + u_4)/4$'))

        q4 = self.p1
        for i in range(1,4):
           q1 = self.p1 + (-2.5 + i, -2.1)
           q2 = q1 + (0.4, 0.5)
           q2 = q4 + (0, -0.5)
           q3 = q4 + (0, -1.3)
           p.add(Text(q1, '$u_' + str(i) + '$'))
        p.add(Arc(self.p1, 0.6, -30, 330, linewidth=0.07, arrow="->", arrowinset=0, arrowsize="3pt 3"))
        
        for i in range(2,5):
           q1 = Point(6.5 + i, 2.5)
           q2 = q1 + (0, 2)
           q4 = self.p2 + (-3.5 + i, -2)
           q3 = q4 + (0, -2)
           p.add(Text(q4, '$u_' + str(i) + '$'))
        p.add(Arc(self.p2, 0.6, -30, 330, linewidth=0.07, arrow="->", arrowinset=0, arrowsize="3pt 3"))
        
        super().plot(p)
        
class Scene4(Scene0):
    def __init__(self, L, H):
        super().__init__(L, H)
        
    def plot(self, p):
        for i in range(1,4):
           q1 = self.p1 + (-2.5 + i, -2.1)
           p.add(Text(q1, '$u_' + str(i) + '$'))
           
        q1 = self.p1
        q2 = q1 + (-3, -1)
        q3 = q2 + (0, -1)
        q4 = q3 + (1.4, -0.8)
        p.add(Bezier(q1, q2, q3, q4, \
                     linecolor='red', arrow='->', linewidth=0.1, arrowsize=0.4, \
                     showpoints='false'))
        p.add(Text(self.p1 + (-1.5, -3.0), '$v_2$'))


        for i in range(2,5):
           q1 = Point(6.5 + i, 2.5)
           q2 = q1 + (0, 2)
           q4 = self.p2 + (-3.5 + i, -2)
           q3 = q4 + (0, -2)
           p.add(Text(q4, '$u_' + str(i) + '$'))
        q1 = self.p2
        q2 = q1 + (-3, -1)
        q3 = q2 + (0, -1)
        q4 = q3 + (1.4, -0.8)
        p.add(Bezier(q1, q2, q3, q4, \
                     linecolor='red', arrow='->', linewidth=0.1, arrowsize=0.4, \
                     showpoints='false'))
        p.add(Text(self.p2 + (-1.5, -3.0), '$v_3$'))
        
        super().plot(p)
        
class Scene5(Scene0):
    def __init__(self, L, H):
        super().__init__(L, H)
        
    def plot(self, p):
        q4 = self.p1
        for i in range(1,4):
           q1 = self.p1 + (-2.5 + i, -2.1)
           q2 = q1 + (0.4, 0.5)
           q2 = q4 + (0, -0.5)
           q3 = q4 + (0, -1.3)
           p.add(Text(q1, '$u_' + str(i) + '$'))
        p.add(Text(self.p1 + (-1.5, -3), '$v_2$'))
        

        for i in range(2,5):
           q1 = Point(6.5 + i, 2.5)
           q2 = q1 + (0, 2)
           q4 = self.p2 + (-3.5 + i, -2)
           q3 = q4 + (0, -2)
           p.add(Text(q4, '$u_' + str(i) + '$'))
        p.add(Text(self.p2 + (-1.5, -3), '$v_3$'))
        
        super().plot(p)
        q1 = self.p1 + (-1., -3.3)
        q2 = q1 + (1, -2)
        q3 = q2 + (0, -1)
        q4 = q3 + (5, -0.8)
        p.add(Bezier(q1, q2, q3, q4, \
                     linecolor='red', arrow='->', linewidth=0.1, arrowsize=0.4, \
                     showpoints='false'))
       
        q1 = self.p2 + (-1., -3.3)
        q2 = q1 + (0, -2)
        q3 = q2 + (0, -1)
        q4 = q3 + (-0.5, -0.6)
        p.add(Bezier(q1, q2, q3, q4, \
                     linecolor='blue', arrow='->', linewidth=0.1, arrowsize=0.4, \
                     showpoints='false'))
       
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

S = Scene5(L, H)

p.open('multithread5.tex')
S.plot(p)
p.close()

