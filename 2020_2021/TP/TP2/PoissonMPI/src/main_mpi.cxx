#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <mpi.h>

#include "parameters.hxx"
#include "values.hxx"
#include "scheme.hxx"
#include "timer.hxx"
#include "os.hxx"

double cond_ini(double x, double y, double z)
{
  x -= 0.5;
  y -= 0.5;
  z -= 0.5;
  if (x*x+y*y+z*z < 0.1)
    return 1.0;
  else
    return 0.0;
}

double force(double x, double y, double z)
{
  if (x < 0.3)
    return 0.0;
  else
    return sin(x-0.5) * exp(- y*y);
}

int main(int argc, char *argv[])
{
  AddTimer("total");
  AddTimer("init");
  AddTimer("calcul");
  AddTimer("other");

  Timer & T_total = GetTimer(0);
  Timer & T_init = GetTimer(1);
  Timer & T_calcul = GetTimer(2);
  Timer & T_other = GetTimer(3);

  T_total.start();

  Parameters Prm(argc, argv);
  if (Prm.help()) return 0;
  std::cout << Prm << std::endl;
  
  int itMax = Prm.itmax();
  int freq = Prm.freq();

  T_init.start();

  Scheme C(Prm, force);

  Values u_0(Prm);
  u_0.boundaries(cond_ini);
  u_0.init(cond_ini);
  C.setInput(u_0);
  T_init.stop();
  std::cout << "\n  temps init "  << std::setw(10) << std::setprecision(6) 
            << T_init.elapsed() << " s\n" << std::endl;

  for (int it=0; it < itMax; it++) {
    if (freq > 0 && it % freq == 0) {
      T_other.start();
      C.getOutput().plot(it);
      T_other.stop();
    }

    T_calcul.start();
    C.iteration();
    T_calcul.stop();

    std::cout << "iteration " << std::setw(5) << it 
              << "  variation " << std::setw(10) << C.variation()
              << "  temps calcul " << std::setw(10) << std::setprecision(6) 
              << T_calcul.elapsed() << " s"
              << std::endl; 
}

  if (freq > 0 && itMax % freq == 0) {
    T_other.start();
    C.getOutput().plot(itMax);
    T_other.stop();
    }

  T_total.stop();

  std::cout << "\n" << std::setw(26) << "temps total" 
            << std::setw(10) << T_total.elapsed() << " s\n" << std::endl;

  int id = 0;

  std::string s = Prm.resultPath();
  mkdir_p(s.c_str());
  s += "/temps_";
  s += std::to_string(id) + ".dat";
  std::ofstream f(s.c_str());
  f << id << " " << T_total.elapsed() << std::endl;

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
  if (coord[0] == dim[0]-1) {
    nGlobal_int_max = nGlobal;
    n = nGlobal - n * (dim[0]-1);
  }
  int nGlobal_ext_min = nGlobal_int_min - 1;
  int nGlobal_ext_max = nGlobal_int_max + 1;

  int mGlobal_int_min = 1 + coord[1]*m;
  int mGlobal_int_max = mGlobal_int_min + m - 1;
  if (coord[1] == dim[1]-1) {
    mGlobal_int_max = nGlobal;
    m = nGlobal - m * (dim[1]-1);
  }
  int mGlobal_ext_min = mGlobal_int_min - 1;
  int mGlobal_ext_max = mGlobal_int_max + 1;
  
  xmin = dx * nGlobal_ext_min;
  xmax = dx * nGlobal_ext_max;
  ymin = dy * mGlobal_ext_min;
  ymax = dy * mGlobal_ext_max;
}

int main(int argc, char **argv)
{
  Arguments A;
  A.AddArgument("n", 200);
  A.AddArgument("it", 10);
  A.AddArgument("dt", 0.0001);

  A.Parse(argc, argv);

  int rank, size;

  //M.initMeasure();
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //M.setRank(rank, size);
    
  int dim[2] = {size, 1};
  int period[2] = {0, 0};
  int reorder = 0;
  int coord[2];

  MPI_Comm comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &comm);
  MPI_Comm_rank(comm, &rank);
  MPI_Cart_coords(comm, rank, 2, coord);
  //M.endMeasure("MPI_Init + MPI_Cart");

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

  //M.initMeasure();
  MPI_Finalize();
  //M.endMeasure("MPI_Finalize");
  return 0;
}
