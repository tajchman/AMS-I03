#if defined(_WIN32)
#define _CRT_SECURE_NO_WARNINGS
#include <direct.h>
#elif defined(__unix)
#include <unistd.h>
#endif

#include "parameters.hxx"
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <cstring>

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

Parameters::Parameters(int argc, char **argv) : GetPot(argc, argv)
{
  m_help = options_contain("h") or long_options_contain("help");

  m_nthreads = (*this)("threads", 1);
  m_n[0] = (*this)("n", 400);
  m_n[1] = (*this)("m", 400);
  m_n[2] = (*this)("p", 400);
  m_itmax = (*this)("it", 10);
  double dt_max = 1.5/(m_n[0]*m_n[0]
		       + m_n[1]*m_n[1]
		       + m_n[2]*m_n[2]);
  m_dt = (*this)("dt", dt_max);
  m_output = (*this)("out", -1);

  m_convection = (*this)("convection", 1) == 1;
  m_diffusion = (*this)("diffusion", 1) == 1;
  
  if (!m_help) {
 
    if (m_dt > dt_max)
      std::cerr << "Warning : provided dt (" << m_dt
		<< ") is greater then the recommended maximum (" <<  dt_max
		<< ")" << std::endl;
    
    for (int i=0; i<3; i++) {
      m_dx[i] = m_n[i]>1 ? 1.0/(m_n[i]-1) : 0.0;
      m_di[i] = 1;
      m_imin[i] = 1;
      m_imax[i] = m_n[i]-1;
      if (m_n[i] < 2) {
	m_imin[i]=0; m_imax[i] = 1; m_di[i] = 0;
      }
    }
  
    char buffer[256];
    stime(buffer, 256);
    
    std::ostringstream pth;
    pth << "results"
        << "_n_" << m_n[0] << "x" << m_n[1] << "x" << m_n[2]
        << "_" << buffer << "/";
    m_path = pth.str();

#if defined(_WIN32)
    (void) _mkdir(m_path.c_str());
#else 
    mkdir(m_path.c_str(), 0777);
#endif

    std::ostringstream s;
    s << m_path << "/out.txt";
    m_out = new std::ofstream(s.str().c_str());
  }
  else
    m_out = NULL;
}

bool Parameters::help()
{
  if (m_help) {
    std::cerr << "Usage : ./PoissonOpenMP <list of options>\n\n";
    std::cerr << "Options:\n\n"
              << "-h|--help     : display this message\n"
	      << "threads=<int> : nombre de threads OpenMP"
              << "convection=0/1: convection term (default: 1)\n"
              << "diffusion=0/1 : convection term (default: 1)\n"
              << "n=<int>       : number of internal points in the X direction (default: 400)\n"
              << "m=<int>       : number of internal points in the Y direction (default: 400)\n"
              << "p=<int>       : number of internal points in the Z direction (default: 400)\n"
              << "dt=<real>     : time step size (default : value to assure stable computations)\n"
              << "it=<int>      : number of time steps (default : 10)\n"
              << "out=<int>     : number of time steps between saving the solution on files\n"
              << "                (default : no output)\n\n";
  }
  return m_help;
}

Parameters::~Parameters()
{
  if (!m_help) {
    delete m_out;
  }
}

std::ostream & operator<<(std::ostream &f, const Parameters & p)
{
  f << "Domain :   "
    << "[" << 0 << "," << p.n(0) - 1  << "] x "
    << "[" << 0 << "," << p.n(1) - 1  << "] x "
    << "[" << 0 << "," << p.n(2) - 1  << "]"
    << "\n\n";

  f << "It. max : " << p.itmax() << "\n"
    << "Dt :      " << p.dt() << "\n";

  return f;
}
