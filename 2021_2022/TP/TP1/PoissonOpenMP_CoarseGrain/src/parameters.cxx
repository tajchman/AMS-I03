
#if defined(_WIN32)
#define _CRT_SECURE_NO_WARNINGS
#include <direct.h>
#elif defined(__unix)
#include <unistd.h>
#endif

#if defined(_OPENMP)
   #include <omp.h>
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

Parameters::Parameters(int argc, char ** argv) : Arguments(argc, argv)
{
  m_command = argv[0];
  m_help = options_contains("h") || options_contains("help");

  if (m_help) return;

#if defined(_OPENMP)
  m_nthreads = Get("threads", 0);

  if (m_nthreads<1) {
    const char * omp_var = std::getenv("OMP_NUM_THREADS");
    if (omp_var) m_nthreads = strtol(omp_var, NULL, 10);
    m_nthreads = Get("threads", m_nthreads);
  }

  if (m_nthreads<1)
    m_nthreads=1;

  omp_set_num_threads(m_nthreads);
#endif

  m_n[0] = Get("n0", 401);
  m_n[1] = Get("n1", 401);
  m_n[2] = Get("n2", 401);
  m_itmax = Get("it", 10);

  double d;
  double dt_max = 0.1/(m_n[0]*m_n[0]);
  d = 0.1/(m_n[1]*m_n[1]);
  if (dt_max > d) dt_max = d;
  d = 0.1/(m_n[2]*m_n[2]);
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
      m_dx[i] = m_n[i]>1 ? 1.0/(m_n[i]-1) : 0.0;
      m_imin[i] = 1;
      m_imax[i] = m_n[i]-1;
      m_xmax[i] = 1.0;
    }
  }
 
  #ifdef _OPENMP
  int nt = m_nthreads;
  #else
  int nt = 1;
  #endif

  int idecoupe = -1, iT, maxn=-1;
  for (int i=0; i<3; i++) {
    m_imin_local[i].resize(nt);
    m_imax_local[i].resize(nt);

    for (iT = 0; iT < nt; iT++) 
    {
       m_imin_local[i][iT] = m_imin[i];
       m_imax_local[i][iT] = m_imax[i];
    }
    if (m_imax[i] - m_imin[i] > maxn) {
      idecoupe = i;
      maxn = m_imax[i] - m_imin[i];
    }
  }

  maxn++;
  int di = maxn/nt;
  
  int i0 = m_imin[idecoupe];
  for (iT=0; iT < nt;iT++) {
    m_imin_local[idecoupe][iT] = i0;
    m_imax_local[idecoupe][iT] = i0 + di - 1;
    i0 += di;
  }
  m_imax_local[idecoupe][nt-1] = m_imax[idecoupe];
 
#ifdef DEBUG
  for (iT=0; iT<nt; iT++) {
    std::cerr << "Thread " << iT;
    for (int i=0; i < 3; i++)
      std::cerr<< "  [" << m_imin_local[i][iT] << "," 
               << m_imax_local[i][iT] <<"]";
    std::cerr << std::endl;
  }
#endif
}

bool Parameters::help()
{
  if (m_help) {
    std::cerr << "Usage : ./PoissonOpenMP <list of options>\n\n";
    std::cerr << "Options:\n\n"
              << "-h|--help     : display this message\n"
#ifdef _OPENMP
              << "threads=<int> : number of threads OpenMP"
#endif
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

  f << "  Point indices :   "
    << "[" << p.imin(0) << " ... " << p.imax(0) << "] x "
    << "[" << p.imin(1) << " ... " << p.imax(1) << "] x "
    << "[" << p.imin(2) << " ... " << p.imax(2) << "]\n\n";

#ifdef _OPENMP
  f << p.nthreads() << " thread(s)\n";
#endif
  f << "It. max :  " << p.itmax() << "\n"
    << "Dt :       " << p.dt() << "\n"
    << "Results in " << p.resultPath() << std::endl;

  return f;
}

