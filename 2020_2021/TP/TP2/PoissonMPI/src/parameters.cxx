
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

Parameters::Parameters(int argc, char ** argv, int size, int rank) : Arguments(argc, argv)
{
  m_size = size;
  m_rank = rank;

  int dim[3] = {size, 1, 1};
  int period[3] = {0, 0, 0};
  int reorder = 0;
  int coord[3];

  MPI_Cart_create(MPI_COMM_WORLD, 3, dim, period, reorder, &m_comm);
  MPI_Comm_rank(m_comm, &rank);
  MPI_Cart_coords(m_comm, rank, 3, coord);
 
  m_help = options_contains("h") || options_contains("help");

  m_command = argv[0];

  m_n_global[0] = Get("n0", 400);
  m_n_global[1] = Get("n1", 400);
  m_n_global[2] = Get("n2", 400);
  m_itmax = Get("it", 10);

  double d;
  double dt_max = 0.1/(m_n_global[0]*m_n_global[0]);
  d = 0.1/(m_n_global[1]*m_n_global[1]);
  if (dt_max > d) dt_max = d;
  d = 0.1/(m_n_global[2]*m_n_global[2]);
  if (dt_max > d) dt_max = d;
 
  m_dt = Get("dt", dt_max);
  m_freq = Get("out", -1);
  
  if (!m_help) {
 
    m_path = Get("path", ".");
    if (m_path != ".") 
       mkdir_p(m_path.c_str());

    if (m_dt > dt_max)
      std::cerr << "Warning : provided dt (" << m_dt
                << ") is greater then the recommended maximum (" <<  dt_max
                << ")" << std::endl;
    
    for (int i=0; i<3; i++) {
      m_xmin[i] = 0.0;
      m_dx[i] = m_n_global[i]>1 ? 1.0/(m_n_global[i]+1) : 0.0;
      m_n[i] = m_n_global[i]/dim[i];
  
      int nGlobal_int_min = 1 + coord[i]*m_n[i];
      int nGlobal_int_max = nGlobal_int_min + m_n[i] - 1;
      if (coord[i] == dim[i]-1) {
        nGlobal_int_max = m_n_global[i];
        m_n[i] = m_n_global[i] - m_n[i] * (dim[0]-1);
      }
      int nGlobal_ext_min = nGlobal_int_min - 1;
      int nGlobal_ext_max = nGlobal_int_max + 1;
      m_imin[i] = 1;
      m_imax[i] = m_n[i]-1;

      m_xmin[i] = m_dx[i] * nGlobal_ext_min;
      m_xmax[i] = m_dx[i] * nGlobal_ext_max;
    }
  }
}

bool Parameters::help()
{
  if (m_help) {
    std::cerr << "Usage : ./PoissonOpenMP <list of options>\n\n";
    std::cerr << "Options:\n\n"
              << "-h|--help     : display this message\n"
              << "n0=<int>       : number of points in the X direction (default: 400)\n"
              << "n1=<int>       : number of points in the Y direction (default: 400)\n"
              << "n2=<int>       : number of points in the Z direction (default: 400)\n"
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
    << "[" << 0 << "," << p.n(0) - 1  << "] x "
    << "[" << 0 << "," << p.n(1) - 1  << "] x "
    << "[" << 0 << "," << p.n(2) - 1  << "]\n";

  f << "It. max :  " << p.itmax() << "\n"
    << "Dt :       " << p.dt() << "\n"
    << "Results in " << p.resultPath() << std::endl;

  return f;
}

