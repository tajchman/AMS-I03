
#if defined(_WIN32)
#define _CRT_SECURE_NO_WARNINGS
#include <direct.h>
#elif defined(__unix)
#include <unistd.h>
#endif

#include "os.hxx"
#include "arguments.hxx"
#include "parameters.hxx"
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <array>

#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <mpi.h>

void stime(char * buffer, int size)
{
  time_t curtime;
  struct tm *loctime;

  /* Get the current time. */
  curtime = time (NULL);

  /* Convert it to local time representation. */

  loctime = localtime (&curtime);
  strftime(buffer, size, "%F_%Hh%Mm%Ss", loctime);

}

Parameters::Parameters(int argc, char ** argv, int size, int rank)
  : Arguments(argc, argv), m_neighbour{-1, -1, -1, -1, -1, -1}
{
  m_command = argv[0];
  m_help = options_contains("h") || options_contains("help");

  if (m_help) return;

  m_size = size;
  m_rank = rank;

  int dim[3] = {size, 1, 1};
  int period[3] = {0, 0, 0};
  int reorder = 0;
  std::array<int, 3> coord, coord2;

  MPI_Cart_create(MPI_COMM_WORLD, 3, dim, period, reorder, &m_comm);
  MPI_Cart_coords(m_comm, rank, 3, &(coord[0]));
  std::cout << " " << coord[0] << " " << coord[1] << " " << coord[2] << std::endl;

  for (int i=0; i<3; i++) {
    if (coord[i] > 0) {
      coord2 = coord;
      coord2[i]--;
      MPI_Cart_rank(m_comm, &(coord2[0]), &m_neighbour[2*i]);
    }
    if (coord[i] < dim[i]-1) {
      coord2 = coord;
      coord2[i]++;
      MPI_Cart_rank(m_comm, &(coord2[0]), &m_neighbour[2*i+1]);
    }
  }

  m_n_global[0] = Get("n0", 21);
  m_n_global[1] = Get("n1", 21);
  m_n_global[2] = Get("n2", 21);
  m_itmax = Get("it", 10);

  double d;
  double dt_max = 0.1/(m_n_global[0]*m_n_global[0]);
  d = 0.1/(m_n_global[1]*m_n_global[1]);
  if (dt_max > d) dt_max = d;
  d = 0.1/(m_n_global[2]*m_n_global[2]);
  if (dt_max > d) dt_max = d;

  m_dt = Get("dt", dt_max);
  m_freq = Get("out", -1);

  m_path = Get("path", ".");
  if (m_path != ".")
     mkdir_p(m_path.c_str());

  if (m_dt > dt_max)
    std::cerr << "Warning : provided dt (" << m_dt
              << ") is greater then the recommended maximum (" <<  dt_max
              << ")" << std::endl;

  for (int i=0; i<3; i++) {
    m_dx[i] = m_n_global[i]>1 ? 1.0/(m_n_global[i]-1) : 0.0;

    int n = (m_n_global[i]-2)/dim[i];
    int nGlobal_int_min = 1 + coord[i]*n;
    int nGlobal_int_max;
    if (coord[i] < dim[i]-1) {
      nGlobal_int_max = nGlobal_int_min + n - 1;
    }
    else {
      nGlobal_int_max = m_n_global[i] - 2;
    }

    int nGlobal_ext_min = nGlobal_int_min - 1;
    int nGlobal_ext_max = nGlobal_int_max + 1;
    m_imin[i] = 1;
    m_imax[i] = nGlobal_int_max - nGlobal_int_min + 1;

    m_xmin[i] = m_dx[i] * nGlobal_ext_min;
    m_xmax[i] = m_dx[i] * nGlobal_ext_max;

    std::cout << "global_int[" << i << "] : " << nGlobal_int_min << " " << nGlobal_int_max << std::endl;
    std::cout << "global_ext[" << i << "] : " << nGlobal_ext_min << " " << nGlobal_ext_max << std::endl;
  }
}

bool Parameters::help()
{
  if (m_help) {
    std::cerr << "Usage : ./PoissonOpenMP <list of options>\n\n";
    std::cerr << "Options:\n\n"
              << "-h|--help     : display this message\n"
              << "n0=<int>       : number of points in the X direction (default: 401)\n"
              << "n1=<int>       : number of points in the Y direction (default: 401)\n"
              << "n2=<int>       : number of points in the Z direction (default: 401)\n"
              << "dt=<real>     : time step size (default : value to assure stable computations)\n"
              << "it=<int>      : number of time steps (default : 10)\n"
              << "out=<int>     : number of time steps between saving the solution on files\n"
              << "                (default : no output)\n"
              << "path=<string> : results directory (default : '.')\n\n";
  }
  return m_help;
}

std::ostream & operator<<(std::ostream &f, const Parameters & p)
{
  f << "Domain :   "
    << "[" << p.xmin(0) << ", " << p.xmax(0) << "] x "
    << "[" << p.xmin(1) << ", " << p.xmax(1) << "] x "
    << "[" << p.xmin(2) << ", " << p.xmax(2) << "]\n";

  f << "Interior points :   "
    << "[" << p.imin(0) << ", ..., " << p.imax(0) << "] x "
    << "[" << p.imin(1) << ", ..., " << p.imax(1) << "] x "
    << "[" << p.imin(2) << ", ..., " << p.imax(2) << "]\n";

  f << "It. max :  " << p.itmax() << "\n"
    << "Dt :       " << p.dt() << "\n"
    << "Results in " << p.resultPath() << std::endl;

  return f;
}
