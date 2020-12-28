
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

#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>

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
  m_help = options_contains("h") || options_contains("help");

  m_command = argv[0];

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

  m_n[0] = Get("n0", 400);
  m_n[1] = Get("n1", 400);
  m_n[2] = Get("n2", 400);
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
    }
  }
}

bool Parameters::help()
{
  if (m_help) {
    std::cerr << "Usage : ./PoissonOpenMP <list of options>\n\n";
    std::cerr << "Options:\n\n"
              << "-h|--help     : display this message\n"
#ifdef _OPENMP
              << "threads=<int> : nombre de threads OpenMP"
#endif
              << "convection=0/1: convection term (default: 1)\n"
              << "diffusion=0/1 : convection term (default: 1)\n"
              << "n1=<int>       : number of points in the X direction (default: 400)\n"
              << "n2=<int>       : number of points in the Y direction (default: 400)\n"
              << "n3=<int>       : number of points in the Z direction (default: 400)\n"
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

#ifdef _OPENMP
  f << p.nthreads() << " thread(s)\n";
#endif
  f << "It. max :  " << p.itmax() << "\n"
    << "Dt :       " << p.dt() << "\n"
    << "Results in " << p.resultPath() << std::endl;

  return f;
}

