from figures import Point, Text, Rectangle, Circle, Line, Lines, Bezier, pstricks

L = 16
H = 10

class Scene0:
   def __init__(self, L, H):
      self.p1 = Point(L/2., H-1)
      pass

   def plot0(self, p):
      p.add(Text(self.p1 + (1.2, 0), r'C\oe ur'))
      p.add(Text(self.p1 + (1.7, -2), 'Mémoire'))
      p.add(Text(self.p1 + (1.7, -2.7), 'cache'))
      p.add(Text((0.3, 2.3), r"M\'emoire"))
 

   def plot(self, p):
      p.add(Rectangle((0,0), (L,3)))
      p.add(Circle(self.p1 - (0,0.1), 1.0))
      p.add(Rectangle(self.p1 + (-1.5,-5.5), self.p1 + (1.5,-1.5)))
      p.add(Line(self.p1 + (-1.5,-2.5), self.p1 + (1.5,-2.5)))
      p.add(Line(self.p1 + (-1.5,-3.5), self.p1 + (1.5,-3.5)))
      p.add(Line(self.p1 + (-1.5,-4.5), self.p1 + (1.5,-4.5)))
      p.add(Line(self.p1 + (0,-1.5), self.p1 + (0,-5.5), linestyle='dashed'))
      p.add(Text((5.15,0.8), '$v$'))
      p.add(Text((5.15,2.0), '$u$'))
      for i in range(7,12):
         k = 6 + (i-6)*1.1
         p.add(Text((k-1,0.8), '$v_' + str(i-7) + '$'))
         p.add(Text((k-1,2.0), '$u_' + str(i-7) + '$'))
         p.add(Line((k,0.5), (k,1.5)))
         p.add(Line((k,1.7), (k,2.7)))
      p.add(Text((11.6,0.8), r'\ldots'))
      p.add(Text((11.6,2.0), r'\ldots'))
      p.add(Rectangle((6,0.5), (12.6,1.5)))
      p.add(Rectangle((6,1.7), (12.6,2.7)))

   def plot2(self, p):
      p.add(Text(self.p1 + (-6, -3.1), 'de cache'))
      p.add(Text(self.p1 + (-6, -2.4), 'Lignes'))
      p.add(Lines((self.p1 + (-5.9,-3.2), self.p1 + (-3,-3.2), self.p1 + (-1.5,-2.0)),
               linewidth=0.07, arrow = "->"))
      p.add(Line(self.p1 + (-3,-3.2), self.p1 + (-1.5,-3.0), linewidth=0.07, arrow = "->"))
      p.add(Line(self.p1 + (-3,-3.2), self.p1 + (-1.5,-4.0), linewidth=0.07, arrow = "->"))
      p.add(Line(self.p1 + (-3,-3.2), self.p1 + (-1.5,-5.0), linewidth=0.07, arrow = "->"))

class Scene1(Scene0):
    def __init__(self, L, H):
        super().__init__(L, H)

    def plot(self, p):
        q1 = Point(6.1, 2.1)
        q4 = Point(6.6, 7.0)
        q2 = (q1*2 + q4)/3. + (-1,0)
        q3 = (q1 + q4*2)/3. + (-1,0)
        p.add(Bezier(q1 , q2, q3, q4 , \
                    linecolor='red', arrow='->', linewidth=0.08, arrowsize=0.4))
        q1 = Point(10.1, 2.6)
        q4 = Point(9.6, 6.0)
        q2 = (q1*2 + q4)/3. + (1,0)
        q3 = (q1 + q4*2)/3. + (1,0)
        p.add(Bezier(q1 , q2, q3, q4 , \
                    linecolor='blue', arrow='->', linewidth=0.08, arrowsize=0.4))
        q1 = Point(6.1, 0.9)
        q4 = Point(6.6, 5.0)
        q2 = (q1*2 + q4)/3. + (-2,0)
        q3 = (q1 + q4*2)/3. + (-2,0)
        p.add(Bezier(q1 , q2, q3, q4 , \
                    linecolor='violet', arrow='->', linewidth=0.08, arrowsize=0.4))
        q1 = Point(6.7, 6.8)
        p.add(Text(q1, '$u_0$'))
        p.add(Text(q1 + (1.5,0), '$u_1$'))
        q2 = q1 - (0, 1)
        p.add(Text(q2, '$u_2$'))
        p.add(Text(q2 + (1.5,0), '$u_3$'))
        q3 = q2 - (0, 1)
        p.add(Text(q3, '$v_0$'))
        p.add(Text(q3 + (1.5,0), '$v_1$'))
        super().plot(p)
        p.add(Rectangle(Point(6,1.7) + (0.05,0.05), Point(8.2,2.7) - (0.05,0.05), linecolor='red', linewidth=0.08))
        p.add(Rectangle(Point(8.2,1.7) + (0.05,0.05), Point(10.4,2.7) - (0.05,0.05), linecolor='blue', linewidth=0.08))
        p.add(Rectangle(Point(6,0.5) + (0.05,0.05), Point(8.2,1.5) - (0.05,0.05), linecolor='violet', linewidth=0.08))
        p.add(Rectangle(self.p1 + (-1.4,-2.4), self.p1 + (1.4,-1.6), linecolor='red', linewidth=0.08))
        p.add(Rectangle(self.p1 + (-1.4,-3.4), self.p1 + (1.4,-2.6), linecolor='blue', linewidth=0.08))
        p.add(Rectangle(self.p1 + (-1.4,-4.4), self.p1 + (1.4,-3.6), linecolor='violet', linewidth=0.08))

class Scene2(Scene0):
    def __init__(self, L, H):
        super().__init__(L, H)

    def plot(self, p):
        q1 = Point(6.7, 6.8)
        p.add(Text(q1, '$u_0$'))
        p.add(Text(q1 + (1.5,0), '$u_1$'))
        q2 = q1 - (0, 1)
        p.add(Text(q2, '$u_2$'))
        p.add(Text(q2 + (1.5,0), '$u_3$'))
        q3 = q2 - (0, 1)
        p.add(Text(q3, '$v_0$'))
        p.add(Text(q3 + (1.5,0), '$v_1$'))

        q1 = Point(6.5, 7.0)
        q4 = self.p1 + (-0.9, -0.3)
        q2 = (q1*2 + q4)/3. + (-0.7,0)
        q3 = (q1 + q4*2)/3. + (-0.7,0)
        p.add(Rectangle(q1 + (+1.4, -0.4), q1 + (+0.1,0.4), linecolor='red', linewidth=0.08))
        p.add(Bezier(q1 + (0.1, 0), q2, q3, q4 , \
                    linecolor='red', arrow='->', linewidth=0.08, arrowsize=0.4))
        q1 = Point(9.5, 7.0)
        q4 = self.p1 + (0.9, -0.3)
        q2 = (q1*2 + q4)/3. + (0.7,0)
        q3 = (q1 + q4*2)/3. + (0.7,0)
        p.add(Rectangle(q1 + (-1.4, -0.4), q1 + (-0.1,0.4), linecolor='red', linewidth=0.08))
        p.add(Bezier(q1 + (-0.1, 0) , q2, q3, q4 , \
                    linecolor='red', arrow='->', linewidth=0.08, arrowsize=0.4))
        q1 = Point(6.5, 6.0)
        q4 = self.p1 + (-0.9, 0.3)
        q2 = (q1*2 + q4)/3. + (-2,0)
        q3 = (q1 + q4*2)/3. + (-2,0)
        p.add(Rectangle(q1 + (+1.4, -0.4), q1 + (+0.1,0.4), linecolor='red', linewidth=0.08))
        p.add(Bezier(q1 + (0.1, 0), q2, q3, q4 , \
                    linecolor='red', arrow='->', linewidth=0.08, arrowsize=0.4))
        q1 = Point(9.5, 5.0)
        q4 = self.p1 + (0.9, 0.3)
        q2 = (q1*2 + q4)/3. + (2,0)
        q3 = (q1 + q4*2)/3. + (2,0)
        p.add(Rectangle(q1 + (-1.4, -0.4), q1 + (-0.1,0.4), linecolor='blue', linewidth=0.08))
        p.add(Bezier(q1 + (-0.1, 0) , q2, q3, q4 , \
                    linecolor='blue', arrow='<-', linewidth=0.08, arrowsize=0.4))
        super().plot(p)


class Scene3(Scene0):
    def __init__(self, L, H):
        super().__init__(L, H)

    def plot(self, p):
        q1 = Point(6.7, 6.8)
        p.add(Text(q1, '$u_0$'))
        p.add(Text(q1 + (1.5,0), '$u_1$'))
        q2 = q1 - (0, 1)
        p.add(Text(q2, '$u_2$'))
        p.add(Text(q2 + (1.5,0), '$u_3$'))
        q3 = q2 - (0, 1)
        p.add(Text(q3, '$v_0$'))
        p.add(Text(q3 + (1.5,0), '$v_1$'))

        q1 = Point(6.5, 5.0)
        q4 = Point(6, 1.2)
        q2 = (q1*2 + q4)/3. + (-2,0)
        q3 = (q1 + q4*2)/3. + (-2,0)
        p.add(Bezier(q1 , q2, q3, q4 , \
                    linecolor='blue', arrow='->', linewidth=0.08, arrowsize=0.4))

        super().plot(p)

        p.add(Rectangle(Point(6.0,0.5) + (0.05,0.05), Point(8.2,1.5) - (0.05,0.05), linecolor='blue', linewidth=0.08))
        p.add(Rectangle(self.p1 + (-1.4,-4.45), self.p1 + (1.4,-3.55), linecolor='blue', linewidth=0.08))
        


class Scene4(Scene0):
    def __init__(self, L, H):
        super().__init__(L, H)

    def plot(self, p):
        q1 = Point(6.7, 6.8)
        p.add(Text(q1, '$u_0$'))
        p.add(Text(q1 + (1.5,0), '$u_1$'))
        q2 = q1 - (0, 1)
        p.add(Text(q2, '$u_2$'))
        p.add(Text(q2 + (1.5,0), '$u_3$'))
        q3 = q2 - (0, 1)
        p.add(Text(q3, '$v_0$'))
        p.add(Text(q3 + (1.5,0), '$v_1$'))

        super().plot(p)

class Scene5(Scene0):
    def __init__(self, L, H):
        super().__init__(L, H)

    def plot(self, p):
        q1 = Point(6.7, 6.8)
        p.add(Text(q1, '$u_0$'))
        p.add(Text(q1 + (1.5,0), '$u_1$'))
        q2 = q1 - (0, 1)
        p.add(Text(q2, '$u_2$'))
        p.add(Text(q2 + (1.5,0), '$u_3$'))
        q3 = q2 - (0, 1)
        p.add(Text(q3, '$v_2$'))
        p.add(Text(q3 + (1.5,0), '$v_3$'))

        super().plot(p)
        q1 = Point(8, 4.55)
        q4 = Point(8.2, 1.2)
        q2 = (q1*2 + q4)/3. + (-1,0)
        q3 = (q1 + q4*2)/3. + (-1,0)
        p.add(Bezier(q1 , q2, q3, q4 , \
                    linecolor='blue', arrow='<-', linewidth=0.08, arrowsize=0.4))
                    
        p.add(Rectangle(Point(8.2,0.5) + (0.05,0.05), Point(10.4,1.5) - (0.05,0.05), linecolor='blue', linewidth=0.08))
        p.add(Rectangle(self.p1 + (-1.4,-4.45), self.p1 + (1.4,-3.55), linecolor='blue', linewidth=0.08))

class Scene6(Scene0):
    def __init__(self, L, H):
        super().__init__(L, H)

    def plot(self, p):
        q1 = Point(6.7, 6.8)
        p.add(Text(q1, '$u_0$'))
        p.add(Text(q1 + (1.5,0), '$u_1$'))
        q2 = q1 - (0, 1)
        p.add(Text(q2, '$u_2$'))
        p.add(Text(q2 + (1.5,0), '$u_3$'))
        q3 = q2 - (0, 1)
        p.add(Text(q3, '$v_2$'))
        p.add(Text(q3 + (1.5,0), '$v_3$'))

        q1 = Point(9.5, 7.0)
        q4 = self.p1 + (0.9, -0.3)
        q2 = (q1*2 + q4)/3. + (0.7,0)
        q3 = (q1 + q4*2)/3. + (0.7,0)
        p.add(Rectangle(q1 + (-1.4, -0.4), q1 + (-0.1,0.4), linecolor='red', linewidth=0.08))
        p.add(Bezier(q1 + (-0.1, 0) , q2, q3, q4 , \
                    linecolor='red', arrow='->', linewidth=0.08, arrowsize=0.4))
        q1 = Point(6.5, 6.0)
        q4 = self.p1 + (-0.9, -0.3)
        q2 = (q1*2 + q4)/3. + (-1,0)
        q3 = (q1 + q4*2)/3. + (-1,0)
        p.add(Rectangle(q1 + (+1.4, -0.4), q1 + (+0.1,0.4), linecolor='red', linewidth=0.08))
        p.add(Bezier(q1 + (0.1, 0), q2, q3, q4 , \
                    linecolor='red', arrow='->', linewidth=0.08, arrowsize=0.4))
        q1 = Point(9.5, 6.0)
        q4 = self.p1 + (0.9, 0.3)
        q2 = (q1*2 + q4)/3. + (1,0)
        q3 = (q1 + q4*2)/3. + (2,0)
        p.add(Rectangle(q1 + (-1.4, -0.4), q1 + (-0.1,0.4), linecolor='red', linewidth=0.08))
        p.add(Bezier(q1 + (-0.1, 0) , q2, q3, q4 , \
                    linecolor='red', arrow='->', linewidth=0.08, arrowsize=0.4))
        q1 = Point(6.5, 5.0)
        q4 = self.p1 + (-0.9, 0.3)
        q2 = (q1*2 + q4)/3. + (-2,0)
        q3 = (q1 + q4*2)/3. + (-2,0)
        p.add(Rectangle(q1 + (+1.4, -0.4), q1 + (+0.1,0.4), linecolor='blue', linewidth=0.08))
        p.add(Bezier(q1 + (0.1, 0), q2, q3, q4 , \
                    linecolor='blue', arrow='<-', linewidth=0.08, arrowsize=0.4))
        super().plot(p)

class Scene7(Scene0):
    def __init__(self, L, H):
        super().__init__(L, H)

    def plot(self, p):
        q1 = Point(6.7, 6.8)
        p.add(Text(q1, '$u_0$'))
        p.add(Text(q1 + (1.5,0), '$u_1$'))
        q2 = q1 - (0, 1)
        p.add(Text(q2, '$u_2$'))
        p.add(Text(q2 + (1.5,0), '$u_3$'))
        q3 = q2 - (0, 1)
        p.add(Text(q3, '$v_2$'))
        p.add(Text(q3 + (1.5,0), '$v_3$'))

        super().plot(p)
        q1 = Point(8, 4.55)
        q4 = Point(8.2, 1.2)
        q2 = (q1*2 + q4)/3. + (-1,0)
        q3 = (q1 + q4*2)/3. + (-1,0)
        p.add(Bezier(q1 , q2, q3, q4 , \
                    linecolor='blue', arrow='->', linewidth=0.08, arrowsize=0.4))
                    
        p.add(Rectangle(Point(8.2,0.5) + (0.05,0.05), Point(10.4,1.5) - (0.05,0.05), linecolor='blue', linewidth=0.08))
        p.add(Rectangle(self.p1 + (-1.4,-4.45), self.p1 + (1.4,-3.55), linecolor='blue', linewidth=0.08))

p = pstricks(0,0,L, H, font="huge", fontname="ptm")
S = Scene0(L, H)

p.open('sequentiel.tex')
S.plot0(p)
S.plot(p)
S.plot2(p)
p.close()

p.open('sequentiel0.tex')
S.plot(p)
p.close()

S = Scene1(L, H)

p.open('sequentiel1.tex')
S.plot(p)
p.close()

S = Scene2(L, H)

p.open('sequentiel2.tex')
S.plot(p)
p.close()

S = Scene3(L, H)

p.open('sequentiel3.tex')
S.plot(p)
p.close()

S = Scene4(L, H)

p.open('sequentiel4.tex')
S.plot(p)
p.close()

S = Scene5(L, H)

p.open('sequentiel5.tex')
S.plot(p)
p.close()

S = Scene6(L, H)

p.open('sequentiel6.tex')
S.plot(p)
p.close()

S = Scene7(L, H)

p.open('sequentiel7.tex')
S.plot(p)
p.close()
