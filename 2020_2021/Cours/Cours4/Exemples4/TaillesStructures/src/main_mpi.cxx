#include <iostream>
#include <cmath>
#include <iomanip>
#include <mpi.h>
#include <fstream>

#include "values.hxx"
#include "arguments.hxx"

std::ofstream fOut;

double f(double x, double y)
{
  return 1.0 + sin(M_PI * x) * sin(M_PI * y);
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
  int i, imin = 1, imax = u.n()+1;
  int j, jmin = 1, jmax = u.m()+1;
  double xmin = u.xmin(), dx = u.dx(), ymin = u.ymin(), dy = u.dy();

  for (i=imin; i<imax; i++)
    for (j=jmin; j<jmax; j++)
      u(i,j) = f(xmin + i*dx, ymin + j*dy);
}

void boundary(Values & u, double (*f)(double x, double y), 
              bool isbound[4])
{
  int i, imin = 1, imax = u.n()+1;
  int j, jmin = 1, jmax = u.m()+1;
  double xmin = u.xmin(), xmax = u.xmax(), dx = u.dx(),
         ymin = u.ymin(), ymax = u.ymax(), dy = u.dy();

  if (isbound[0]) {
    for (j=jmin; j<jmax; j++)
      u(imin-1, j) = f(xmin, ymin + i*dy);
  }

  if (isbound[1])
    for (j=jmin; j<jmax; j++)
      u(imax,   j) = f(xmax, ymin + i*dy);

  if (isbound[2])
    for (i=imin; i<imax; i++)
      u(i, jmin-1) = f(xmin + i*dx, ymin);
  
  if (isbound[3])
    for (i=imin; i<imax; i++)
      u(i, jmax)   = f(xmin + i*dx, ymax);
  
}

double iteration(Values & v, Values & u, double dt, double (*f)(double x, double y))
{
  int i, imin = 1, imax = u.n()+1;
  int j, jmin = 1, jmax = u.m()+1;
  double xmin = u.xmin(), xmax = u.xmax(), dx = u.dx(),
         ymin = u.ymin(), ymax = u.ymax(), dy = u.dy();

  double lx = 0.5*dt/(dx*dx);
  double ly = 0.5*dt/(dy*dy);

  double du = 0.0;

  for (i=imin; i<imax; i++)
    for (j=jmin; j<jmax; j++) {
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
  Arguments A(argc, argv);

  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::string outName = "out_" + std::to_string(rank) + ".txt";
  fOut.open(outName);

  int nGlobal = A.Get("n", 50);
  int iT, nT = A.Get("it", 50);
  double dT = A.Get("dt", 0.25/(nGlobal*nGlobal)), du;

  int n = nGlobal/size;
  int m = nGlobal;
  double xmin = 0;
  double xmax = 1;
  double ymin = double(rank) / size;
  double ymax = double(rank+1) / size;

  if (rank == size-1)
     n = nGlobal - n * (size-1);

  Values U(n, m, xmin, xmax, ymin, ymax);
  
  init(U, u0);

  bool isbound[] = {true, true, true, true};
  if (rank > 0) 
     isbound[0] = false;
  if (rank < size - 1) 
     isbound[1] = false;

  boundary(U, g, isbound);

  Values V = U;

  if (n < 10)
    fOut << U << std::endl;
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

  MPI_Finalize();
  return 0;
}
