import math

class Point:
    def __init__(self, x, y):
        self.c = [x,y]

    def __add__(self, p):
        return Point(self[0] + p[0], self[1] + p[1])
    
    def __sub__(self, p):
        return Point(self[0] - p[0], self[1] - p[1])
    
    def __div__(self, a):
        return Point(self[0]/a, self[1]/a)
    
    def __mul__(self, a):
        return Point(self[0]*a, self[1]*a)

    def __getitem__(self, k):
        return self.c[k]
    
    def __setitem__(self, k, item):
        self.c[k] = item

    def __str__(self):
        return ' (' + str(self.c[0]) + ',' + str(self.c[1]) + ')'

    def norm(self):
        l = math.sqrt(self.c[0]**2 + self.c[1]**2)
        if l < 1e-10:
            l = 1.0
        return Point(self.c[0]/l, self.c[1]/l)
    
class grObject:
    def __init__(self):
        self.center = Point(0,0)
        self.prefix = ""
        self.body = ""
        self.name = ""
        pass

    def anchor(self, delta):
        return (self.center[0] + delta[0], 
                self.center[1] + delta[1])

    def setPrefix(self, args):
        s = ""
        for attr in ["linewidth", 
                     "linestyle", 
                     "fillcolor", 
                     "fillstyle", 
                     "linecolor", 
                     "arrowsize",
                     "framearc",
                     "orientation",
                     "dotstyle",
                     "dotscale",
                     "arrowinset",
                     "arrowsize"]:
            if attr in args:
                if s == "":
                    s += "["
                else:
                    s += ","
                    pass
                if not attr == "orientation":
                    s += attr + "="
                s += str(args[attr])
                pass
            pass
        if not s == "":
            s += "]"
        if "arrow" in args:
            s += "{" + str(args["arrow"]) + "}"
        self.prefix = s;
        pass
    
    def listOfPoints(self, p):
        s = ""
        for q in p:
            s += "(" + str(q[0]) + "," + str(q[1]) + ")"
            pass
        return s

    def getCenter(self, p):
        c0 = 0.0
        c1 = 0.0
        for q in p:
            c0 += q[0]
            c1 += q[1]
            pass
        if len(p) > 0:
            c0 /= len(p);
            c1 /= len(p);
            pass
        return Point(c0, c1)

    def __str__(self):
        s = self.name;
        s += self.prefix;
        s += self.body;
        return s

    pass

class Line(grObject):
    def __init__(self, p1, p2, **args):
        grObject.__init__(self)
        self.name = "\\psline"
        self.setPrefix(args)
        p = [p1, p2]
        self.body = self.listOfPoints(p);
        self.center = self.getCenter(p);
        pass
    pass

class Lines(grObject):
    def __init__(self, p, **args):
        grObject.__init__(self)
        self.name = "\\psline"
        self.setPrefix(args)
        self.body = self.listOfPoints(p);
        self.center = self.getCenter(p);
        pass
    pass

class Polygon(grObject):
    def __init__(self, p, **args):
        grObject.__init__(self)
        self.name = "\\pspolygon"
        self.setPrefix(args)
        self.body = self.listOfPoints(p);
        self.center = self.getCenter(p);
        pass
    pass

class Circle(grObject):
    def __init__(self, c, r, **args):
        grObject.__init__(self)
        self.name = "\\pscircle"
        self.setPrefix(args)
        self.body = self.listOfPoints([c]) + "{" + str(r) + "}";
        self.center = c;
        pass
    pass

class Dot(grObject):
    def __init__(self, c, **args):
        grObject.__init__(self)
        self.name = "\\psdot"
        self.setPrefix(args)
        self.body = self.listOfPoints([c]);
        self.center = c;
        pass
    pass

class Text(grObject):
    def __init__(self, p, text, **args):
        grObject.__init__(self)
        self.name = "\\rput[Bl]"
        self.setPrefix(args)
        self.center = p;
        self.body = "{0}" + self.listOfPoints([p]) + \
            "{\psframebox*{" + text + "}}"
        pass
    pass

class Rectangle(grObject):
    def __init__(self, p1, p2, **args):
        grObject.__init__(self)
        self.name = "\\psframe"
        self.setPrefix(args)
        p = [p1, p2]
        self.body = self.listOfPoints(p);
        self.center = self.getCenter(p);
        pass
    pass

class Curve(grObject):
    def __init__(self, p1, p2, p3, p4, **args):
        grObject.__init__(self)
        self.name = "\\pscurve"
        self.setPrefix(args)
        p = [p1, p2, p3, p4]
        self.body = self.listOfPoints(p);
        self.center = self.getCenter(p);
        pass
    pass
  
class Ellipse(grObject):
    def __init__(self, p, r1, r2, **args):
        grObject.__init__(self)
        self.name = "\\psellipse"
        self.setPrefix(args)
        self.body = self.listOfPoints([p]) + "(" + str(r1) +","+ str(r2) + ")";
        self.center = p;
        pass
    pass

class pstricks:
    
    def __init__(self, oX, oY, dimX, dimY, font="small", fontname="phv"):
        self.oX = oX
        self.oY = oY
        self.dimX = dimX
        self.dimY = dimY
        self.font = font
        self.fontname = fontname
        self.f = None
        pass
    
    def open(self, fileName):
        self.f = open(fileName, 'w')
        s = """
\\documentclass[10pt]{standalone}
\\usepackage[T1]{fontenc}
\\usepackage[utf8]{inputenc}
\\usepackage{listings}
\\usepackage{pstricks}
\\usepackage{auto-pst-pdf}
\\usepackage{pst-plot}
\\pagestyle{empty}
\\begin{document}
\\fontfamily{%s}\\selectfont
\\%s
\\boldmath
\\begin{pspicture}(%f,%f)(%f,%f)
""" % (self.fontname, self.font, self.oX, self.oY, self.dimX, self.dimY)
        print >> self.f, s
        pass
    
    def add(self, obj):
        if not self.f is None:
            print >> self.f, str(obj)
        pass

    def adds(self, objs):
        if not self.f is None:
            for obj in objs:
                print >> self.f, str(obj)
        pass
      
    def close(self):
        if not self.f is None:
            print >> self.f, \
                "\\end{pspicture}\n", \
                "\\end{document}"
            pass
        pass

    def parameter(self, name, value):
        if self.f:
            self.f.write('\\psset{%s=%s}\n' % (name, value))
            pass
        pass
   
