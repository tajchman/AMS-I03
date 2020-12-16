#include <iostream>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <mpi.h>

#include "values.hxx"
#include "arguments.hxx"
#include "memory_used.hxx"

std::ofstream fOut;
MemoryUsed M;


double f(double x, double y)
{
  return cos(M_PI * (x-0.5)) * cos(M_PI * (y-0.5));
}
double u0(double x, double y)
{
  return 0.0;
}
double g(double x, double y)
{
  return 1.0;
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

void boundary(Values & u, double (*g)(double x, double y), int neighbour[4])
{
  int i, imin_ext = 0, imax_ext = u.n()+1;
  int j, jmin_ext = 0, jmax_ext = u.m()+1;
  double xmin = u.xmin(), xmax = u.xmax(), dx = u.dx(),
         ymin = u.ymin(), ymax = u.ymax(), dy = u.dy();

  if (neighbour[0] < 0)
    for (j=jmin_ext; j<=jmax_ext; j++)
      u(imin_ext, j) = g(xmin, ymin + j*dy);

  if (neighbour[1] < 0)
    for (j=jmin_ext; j<=jmax_ext; j++)
      u(imax_ext, j) = g(xmax, ymin + j*dy);

  if (neighbour[2] < 0)
    for (i=imin_ext; i<=imax_ext; i++)
      u(i, jmin_ext) = g(xmin + i*dx, ymin);
  
  if (neighbour[3] < 0)
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

  double du, du_local = 0.0;

  for (i=imin_int; i<=imax_int; i++)
    for (j=jmin_int; j<=jmax_int; j++) {
      v(i,j) = u(i,j) 
        + lx * (-2*u(i,j) + u(i+1, j) + u(i-1, j))
        + ly * (-2*u(i,j) + u(i, j-1) + u(i, j+1))
        + dt * f(xmin + i*dx, ymin + j*dy);
      du_local += std::abs(v(i,j) - u(i,j));
    }

//  M.initMeasure(); 
  MPI_Allreduce(&du_local, &du, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//  M.endMeasure("MPI_Allreduce");

  return du;
}

void synchronize(Values & u, 
                 int neighbour[4])
{
  int i, imin_ext = 0, imax_ext = u.n()+1;
  int j, jmin_ext = 0, jmax_ext = u.m()+1;
  int ni = imax_ext - imin_ext + 1;
  int nj = jmax_ext - jmin_ext + 1;
  MPI_Status status;

  if (neighbour[0] >= 0) {
    std::vector<double> bufferIn(nj), bufferOut(nj);
    for (j=jmin_ext; j<=jmax_ext; j++)
      bufferOut[j-jmin_ext] = u(imin_ext+1, j);
    
//    M.initMeasure();
    MPI_Sendrecv(bufferOut.data(), nj, MPI_DOUBLE, neighbour[0], 0, 
                 bufferIn.data(),  nj, MPI_DOUBLE, neighbour[0], 0,
                 MPI_COMM_WORLD, &status);
//    M.endMeasure("MPI_Sendrecv");

    for (j=jmin_ext; j<=jmax_ext; j++)
      u(imin_ext, j) = bufferIn[j-jmin_ext];
    }

  if (neighbour[1] >= 0) {
    std::vector<double> bufferIn(nj), bufferOut(nj);
    for (j=jmin_ext; j<=jmax_ext; j++)
      bufferOut[j-jmin_ext] = u(imax_ext-1, j);

//    M.initMeasure();
    MPI_Sendrecv(bufferOut.data(), nj, MPI_DOUBLE, neighbour[1], 0, 
                 bufferIn.data(),  nj, MPI_DOUBLE, neighbour[1], 0,
                 MPI_COMM_WORLD, &status);
//    M.endMeasure("MPI_Sendrecv");

    for (j=jmin_ext; j<=jmax_ext; j++)
      u(imax_ext, j) = bufferIn[j-jmin_ext];
    }

  if (neighbour[2] >= 0) {
    std::vector<double> bufferIn(ni), bufferOut(ni);
    for (i=imin_ext; i<=imax_ext; i++)
      bufferOut[i-imin_ext] = u(i, jmin_ext+1);

//    M.initMeasure();
    MPI_Sendrecv(bufferOut.data(), ni, MPI_DOUBLE, neighbour[2], 0, 
                 bufferIn.data(),  ni, MPI_DOUBLE, neighbour[2], 0,
                 MPI_COMM_WORLD, &status);
//    M.endMeasure("MPI_Sendrecv");

    for (i=imin_ext; i<=imax_ext; i++)
      u(i, jmin_ext) = bufferIn[i-imin_ext];
    }
 
  if (neighbour[3] >= 0) {
    std::vector<double> bufferIn(ni), bufferOut(ni);
    for (i=imin_ext; i<=imax_ext; i++)
      bufferOut[i-imin_ext] = u(i, jmax_ext-1);

//    M.initMeasure();
    MPI_Sendrecv(bufferOut.data(), ni, MPI_DOUBLE, neighbour[3], 0, 
                 bufferIn.data(),  ni, MPI_DOUBLE, neighbour[3], 0,
                 MPI_COMM_WORLD, &status);
//    M.endMeasure("MPI_Sendrecv");

    for (i=imin_ext; i<=imax_ext; i++)
      u(i, jmax_ext) = bufferIn[i-imin_ext];
    }
 
}

void subDomainGeom(MPI_Comm & comm, int &n, int &m, 
                   double &xmin , double& xmax,
                   double &ymin , double& ymax,
                   int nGlobal, int coord[2], int dim[2])
{
  n = nGlobal/dim[0];
  m = nGlobal/dim[1];
  double dx = 1.0/(nGlobal+1), dy = 1.0/(nGlobal+1);
  
  int nGlobal_int_min = 1 + coord[0]*n;
  int nGlobal_int_max = nGlobal_int_min + n - 1;
  int nGlobal_ext_min = nGlobal_int_min - 1;
  int nGlobal_ext_max = nGlobal_int_max + 1;
  if (coord[0] == dim[0]-1) {
    nGlobal_int_max = nGlobal;
    n = nGlobal - n * (dim[0]-1);
  }

  int mGlobal_int_min = 1 + coord[1]*m;
  int mGlobal_int_max = mGlobal_int_min + m - 1;
  int mGlobal_ext_min = mGlobal_int_min - 1;
  int mGlobal_ext_max = mGlobal_int_max + 1;
  if (coord[1] == dim[1]-1) {
    mGlobal_int_max = nGlobal;
    m = nGlobal - m * (dim[1]-1);
  }
  
  xmin = dx * nGlobal_ext_min;
  xmax = dx * nGlobal_ext_max;
  ymin = dy * mGlobal_ext_min;
  ymax = dy * mGlobal_ext_max;
}

int main(int argc, char **argv)
{
  Arguments A;
  A.AddArgument("n", 50);
  A.AddArgument("it", 10);
  A.AddArgument("dt", 0.0001);

  A.Parse(argc, argv);

  int rank, size;

  M.initMeasure();
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  M.setRank(rank, size);
    
  int dim[2] = {size, 1};
  int period[2] = {0, 0};
  int reorder = 0;
  int coord[2];

  MPI_Comm comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &comm);
  MPI_Comm_rank(comm, &rank);
  MPI_Cart_coords(comm, rank, 2, coord);
  M.endMeasure("MPI_Init + MPI_Cart");

  if (A.GetOption("-h") || A.GetOption("--help")) {
    if (rank==0)
      A.Usage();
    MPI_Finalize();
    return 0;
    }

  std::string outName = "out_" + std::to_string(rank) + ".txt";
  fOut.open(outName);

  int nGlobal, iT, nT;
  A.Get("n", nGlobal);
  A.Get("it", nT);

  double dT, du;
  A.Set("dt", 0.25/(nGlobal*nGlobal));
  A.Get("dt", dT);

  if (rank == 0) {
    std::cout << "\nEquation de la chaleur\n\t[" 
              << nGlobal << " x " << nGlobal << "] points intérieurs\n"
              << "\t" << nT << " iterations en temps\n\tdt = " << dT << "\n" << std::endl;
    std::cout << "\tversion parallele sur " << size << " processus\n" << std::endl;
  }

  int n, m;
  double xmin, xmax, ymin, ymax;
  subDomainGeom(comm, n, m, 
                xmin, xmax,
                ymin, ymax,
                nGlobal, coord, dim);

  Values U(n, m, xmin, xmax, ymin, ymax);
  fOut << U.dx() << std::endl;
  init(U, u0);

  int neighbour[] = {-1, -1, -1, -1};
  if (rank > 0) 
     neighbour[0] = rank-1;
  if (rank < size - 1) 
     neighbour[1] = rank+1;

  boundary(U, g, neighbour);

  synchronize(U, neighbour);

  Values V = U;

  if (n < 10) {
    fOut << "Condition initiale" << std::endl;
    fOut << U << std::endl;
  }
  if (rank == 0)
    std::cout << "iteration " << "  variation " << std::endl;
  
  for (iT=0; iT<nT; iT++) 
  {
    du = iteration(V, U, dT, f);
    swap(U, V);
    synchronize(U, neighbour);

    if (n < 10) {
      fOut << "Itération " << iT+1 << std::endl;
      fOut << U << std::endl;
    }
    if (rank == 0)
        std::cout << std::setw(9) << iT << std::setw(12) << du << std::endl;
  }

  M.initMeasure();
  MPI_Finalize();
  M.endMeasure("MPI_Finalize");
  return 0;
}
