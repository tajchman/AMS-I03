
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

#if defined(_OPENMP)
   #include <omp.h>
#endif

#include "timer.hxx"

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

  Timer& T_comm = GetTimer(3);
  T_comm.start();

  int dim[3] = {size, 1, 1};
  int period[3] = {0, 0, 0};
  int reorder = 0;
  std::array<int, 3> coord, coord2;

  MPI_Cart_create(MPI_COMM_WORLD, 3, dim, period, reorder, &m_comm);
  MPI_Cart_coords(m_comm, rank, 3, &(coord[0]));

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
  T_comm.stop();

  m_nthreads = Get("threads", 1);

#if defined(_OPENMP)
  const char * omp_var = std::getenv("OMP_NUM_THREADS");
  if (omp_var) m_nthreads = strtol(omp_var, NULL, 10);
  omp_set_num_threads(m_nthreads);
#else
  m_nthreads = 1;
#endif

  m_n_global[0] = Get("n0", 801);
  m_n_global[1] = Get("n1", 401);
  m_n_global[2] = Get("n2", 401);
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

  if (rank == 0) {
    if (m_dt > dt_max)
      std::cerr << "Warning : provided dt (" << m_dt
        << ") is greater then the recommended maximum (" << dt_max
        << ")" << std::endl;
  }

  for (int i=0; i<3; i++) {
    m_dx[i] = m_n_global[i]>1 ? 1.0/(m_n_global[i]-1) : 0.0;

    int n = (m_n_global[i]-2)/dim[i] + 1;
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

    m_imin_global[i] = nGlobal_ext_min;
    m_imax_global[i] = nGlobal_ext_max;
  }
//Le découpage de chaque domaine MPI en sous-domainestrait ́es chacun par un thread
  int idecoupe = -1, iT, maxn = -1;
  for (int i = 0; i < 3; i++) {
    m_imin_thread[i].resize(m_nthreads);
    m_imax_thread[i].resize(m_nthreads);

    for (iT = 0; iT < m_nthreads; iT++)
    {
      m_imin_thread[i][iT] = m_imin[i];
      m_imax_thread[i][iT] = m_imax[i];
    }
    if (m_imax[i] - m_imin[i] > maxn) {
      idecoupe = i;
      maxn = m_imax[i] - m_imin[i];
    }
  }

  maxn++;
  int di = maxn / m_nthreads;

  int i0 = m_imin[idecoupe], i1;
  for (iT = 0; iT < m_nthreads; iT++) {
    i1 = i0 + di;
    m_imin_thread[idecoupe][iT] = i0;
    m_imax_thread[idecoupe][iT] = i1;
    i0 = i1 + 1;
  }
  m_imax_thread[idecoupe][m_nthreads - 1] = m_imax[idecoupe];
//FINI
#ifdef DEBUG
  for (iT = 0; iT < m_nthreads; iT++) {
    std::cerr << "Thread " << iT;
    for (int i = 0; i < 3; i++)
      std::cerr << "  [" << m_imin_thread[i][iT] << ","
      << m_imax_thread[i][iT] << ")";
    std::cerr << std::endl;
  }
#endif
}

bool Parameters::help()
{
  if (m_rank == 0 && m_help) {
    std::cerr << "Usage : ./PoissonOpenMP <list of options>\n\n";
    std::cerr << "Options:\n\n"
              << "-h|--help     : display this message\n"
              << "threads=<int> : nombre de threads OpenMP\n"
              << "n0=<int>       : number of points in the X direction (default: 801)\n"
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

void sendString(const std::string& str, int dest, int tag, MPI_Comm comm)
{
  unsigned len = str.size();
  MPI_Send(&len, 1, MPI_UNSIGNED, dest, tag, comm);
  if (len != 0)
    MPI_Send(str.data(), len, MPI_CHAR, dest, tag, comm);
}

void recvString(std::string& str, int src, int tag, MPI_Comm comm)
{
  unsigned len;
  MPI_Status s;
  MPI_Recv(&len, 1, MPI_UNSIGNED, src, tag, comm, &s);
  if (len != 0) {
    char * tmp = new char[len+1];
    MPI_Recv(tmp, len, MPI_CHAR, src, tag, comm, &s);
    tmp[len] = '\0';
    str = tmp;
    delete [] tmp;
  }
  else
    str.clear();
}

std::ostream & operator<<(std::ostream &f, const Parameters & p)
{
  std::ostringstream s;
  s << "Process " << p.rank() << "\n";
  s << "  Domain :   "
    << "[" << p.xmin(0) << ", " << p.xmax(0) << "] x "
    << "[" << p.xmin(1) << ", " << p.xmax(1) << "] x "
    << "[" << p.xmin(2) << ", " << p.xmax(2) << "]\n";

  s << "  Point indices :   "
    << "[" << p.imin_global(0) << " ... " << p.imax_global(0) << "] x "
    << "[" << p.imin_global(1) << " ... " << p.imax_global(1) << "] x "
    << "[" << p.imin_global(2) << " ... " << p.imax_global(2) << "]\n\n";

  for (int iT = 0; iT < p.nthreads(); iT++) {
    s << "    Thread " << iT;
    for (int i = 0; i < 3; i++)
      s << "  [" << p.imin_thread(i, iT) << " ... "
         << p.imax_thread(i,iT) << "]";
    s << std::endl;
  }

  std::string message = s.str();

  if (p.rank() == 0) {
    f << p.nthreads() << " thread(s)\n"
      << "It. max :  " << p.itmax() << "\n"
      << "Dt :       " << p.dt() << "\n"
      << "Results in " << p.resultPath() << "\n" << std::endl;
  }
  MPI_Barrier(p.comm());

  if (p.rank() == 0) {
    f << message;
    for (int i = 1; i < p.size(); i++) {
      recvString(message, i, 0, p.comm());
      f << message;
    }
  }
  else {
    sendString(message, 0, 0, p.comm());
  }

  return f;
}
