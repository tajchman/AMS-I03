from figures import *

L = 12
H = 10
p = pstricks(0,0,L, H, font="huge", fontname="ptm")
p.open('ex1a.tex')

p1 = Point(0.5, H)
p2 = p1 + (3, -1.5)
p.add(Rectangle(p1, p2))
p.add(Text(p1 + (0.5, -1), r"C\oe ur"))

p3 = p1 + (0, -2)
p4 = p3 + (4, -2)
p.add(Rectangle(p3, p4))
p.add(Text(p4 + (0.3, 1), r"M\'emoire cache"))

p5 = p3 + (0, -3)
p6 = p5 + (6, -4)
p.add(Rectangle(p5, p6))
p.add(Text(p6 + (0.3, 1), r"M\'emoire"))
p.add(Text(p6 + (0.3, 0.3), r"principale"))

p
p.close()

