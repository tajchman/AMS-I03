
#if defined(_WIN32)
#define _CRT_SECURE_NO_WARNINGS
#include <direct.h>
#elif defined(__unix)
#include <unistd.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

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

Parameters::Parameters(int argc, char ** argv) : GetPot(argc, argv)
{
  m_help = options_contain("h") or long_options_contain("help");

  m_command = (*argv)[0];
  m_help = (*this).search(2, "-h", "--help");

  m_diffusion = (*this)("diffusion", true);
  m_convection = (*this)("convection", true);
  m_nmax[0] = (*this)("n", 100);
  m_nmax[1] = (*this)("m", 100);
  m_nmax[2] = (*this)("p", 100);
  m_itmax = (*this)("it", 10);
  double dt_max = 1.5/(m_nmax[0]*m_nmax[0]
		       + m_nmax[1]*m_nmax[1]
		       + m_nmax[2]*m_nmax[2]);
  m_dt = (*this)("dt", dt_max);
  m_freq = (*this)("out", -1);

  m_convection = (*this)("convection", 1) == 1;
  m_diffusion = (*this)("diffusion", 1) == 1;
  
  if (!m_help) {

#pragma	omp parallel
    {
#pragma omp master
      std::cerr << omp_get_num_threads() << " thread(s)" << std::endl;
    } 

    if (m_dt > dt_max)
      std::cerr << "Warning : provided dt (" << m_dt
                << ") is greater then the recommended maximum (" <<  dt_max
                << ")" << std::endl;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &m_size);

    int i;
    int periods[3], coords[3];
    for (i=0; i<3; i++) {
      m_p[i] = (m_nmax[i] > 1) ? 0 : 1;
      periods[i] = 0;
    }

    // Creation d'une "grille" de processus MPI
    MPI_Dims_create(m_size, 3, m_p);

    MPI_Cart_create(MPI_COMM_WORLD, 3, m_p, periods, 1, &m_comm);
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Cart_coords(m_comm, m_rank, 3, coords);

    for (i=0; i<3; i++) {
      m_neigh[i][0] = MPI_PROC_NULL;
      m_neigh[i][1] = MPI_PROC_NULL;
      MPI_Cart_shift(m_comm, i, 1, &m_neigh[i][0], &m_neigh[i][1]);
    }

    for (i=0; i<3; i++) {
      m_dx[i] = m_nmax[i]>1 ? 1.0/(m_nmax[i]-1) : 0.0;
      m_n[i] = (m_nmax[i]-2)/m_p[i]+2;
      m_p0[i] = (m_n[i]-2) * coords[i];
      if ((coords[i] == m_p[i]-1) && ((m_n[i]-2)*m_p[i] < m_nmax[i])-2)
	m_n[i] += m_nmax[i]-2 - (m_n[i]-2)*m_p[i];

      m_di[i] = 1;
      m_imin[i] = 1;
      m_imax[i] = m_n[i]-1;
      if (m_n[i] < 2) {
	m_imin[i]=0; m_imax[i] = 1; m_di[i] = 0;
      }

      m_xmin[i] = m_dx[i] * m_p0[i];
    }
  }
  m_out = NULL;
}

bool Parameters::help()
{
  if (m_help) {
    std::cerr << "Usage : mpirun -n <nproc> " << m_command << " <list of options>\n\n"
	      << "where <nproc> is the number of MPI processes\n\n";

    std::cerr << "Options:\n\n"
              << "-h|--help     : display this message\n"
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
  if (!m_help)
    MPI_Finalize();

  if (m_out)
    delete m_out;
}

std::ostream & operator<<(std::ostream &f, const Parameters & p)
{
  f << "Process(es)  : " << p.size()
    << " (" << p.p(0) << " x " << p.p(1) << " x " << p.p(2) << ")\n";

  f << "Whole domain :   "
    << "[" << 0 << "," << p.nmax(0) - 1  << "] x "
    << "[" << 0 << "," << p.nmax(1) - 1  << "] x "
    << "[" << 0 << "," << p.nmax(2) - 1  << "]"
    << "\n\n";

  f << "It. max : " << p.itmax() << "\n"
    << "Dt :      " << p.dt() << "\n";

  return f;
}

std::ostream & Parameters::out()
{
  if (not m_out) {

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
  return *m_out;
}
