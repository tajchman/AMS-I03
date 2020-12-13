#include <iostream>
#include <cmath>
#include <iomanip>
#include <fstream>

#include "values.hxx"
#include "arguments.hxx"

std::ofstream fOut;

double f(double x, double y)
{
  return cos(M_PI * (x-0.5)) * cos(M_PI * (y-0.5);
}
double u0(double x, double y)
{
  return 0.0;
}
double g(double x, double y)
{
  return 1.0;
}
double u0(double x, double y)
{
  return 0.0;
}

void init(Values & u, double (*f)(double x, double y))
{
  int i, imin_int = 1, imax_int = u.n();
  int j, jmin_int = 1, jmax_int = u.m();
  double xmin = u.xmin(), dx = u.dx(), ymin = u.ymin(), dy = u.dy();

  for (i=imin_int; i<=imax_int; i++)
    for (j=jmin_int; j<=jmax_int; j++)
      u(i,j) = u0(xmin + i*dx, ymin + j*dy);
}

void boundary(Values & u, double (*g)(double x, double y))
{
  int i, imin_ext = 0, imax_ext = u.n()+1;
  int j, jmin_ext = 0, jmax_ext = u.m()+1;
  double xmin = u.xmin(), xmax = u.xmax(), dx = u.dx(),
         ymin = u.ymin(), ymax = u.ymax(), dy = u.dy();

  for (j=jmin_ext; j<=jmax_ext; j++)
    u(imin_ext, j) = g(xmin, ymin + i*dy);

  for (j=jmin_ext; j<=jmax_ext; j++)
    u(imax_ext, j) = g(xmax, ymin + i*dy);
  
  for (i=imin_ext; i<=imax_ext; i++)
    u(i, jmin_ext) = g(xmin + i*dx, ymin);

  for (i=imin_ext; i<=imax_ext; i++)
    u(i, jmax_ext) = g(xmin + i*dx, ymax);
}

double iteration(Values & v, Values & u, double dt, double (*f)(double x, double y))
{
  int i, imin_int = 1, imax_int = u.n();
  int j, jmin_int = 1, jmax_int = u.m();
  double xmin = u.xmin(), xmax = u.xmax(), dx = u.dx(),
         ymin = u.ymin(), ymax = u.ymax(), dy = u.dy();

  double lx = 0.5*dt/(dx*dx);
  double ly = 0.5*dt/(dy*dy);

  double du = 0.0;

  for (i=imin_int; i<=imax_int; i++)
    for (j=jmin_int; j<=jmax_int; j++) {
      v(i,j) = u(i,j) 
        + lx * (-2*u(i,j) + u(i+1, j) + u(i-1, j))
        + ly * (-2*u(i,j) + u(i, j-1) + u(i, j+1))
        + dt * f(xmin + i*dx, ymin + j*dy);
      du += std::abs(v(i,j) - u(i,j));
    }
  return du;
}

int main(int argc, char **argv)
{
  Arguments A;
  A.AddArgument("n", 50);
  A.AddArgument("it", 10);
  A.AddArgument("dt", 0.0001);

  A.Parse(argc, argv);

  if (A.GetOption("-h") || A.GetOption("--help")) {
    A.Usage();
    return 0;
  }

  std::string outName = "out.txt";
  fOut.open(outName);

  int n, iT, nT;
  A.Get("n", n);
  A.Get("it", nT);

  double dT, du;
  A.Set("dt", 0.25/(n*n));
  A.Get("dt", dT);

  std::cout << "\nEquation de la chaleur\n\t[" << n << " x " << n << "] points intérieurs\n"
            << "\t" << nT << " iterations en temps\n\tdt = " << dT << "\n" << std::endl;
  std::cout << "\tversion sequentielle\n" << std::endl;
  
  Values U(n,n), V(n,n);
  
  init(U, u0);
  boundary(U, g);
  V = U;

  if (n < 10)
    std::cout << U << std::endl;
  else
    std::cout << "iteration " << "  variation " << std::endl;

  for (iT=0; iT<nT; iT++) 
  {
    du = iteration(V, U, dT, f);
    swap(U, V);

    if (n < 10)
      std::cout << U << std::endl;

    std::cout << std::setw(9) << iT << std::setw(12) << du << std::endl;
  }
  return 0;
}
