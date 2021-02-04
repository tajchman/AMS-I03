
#if defined(_WIN32)
#define _CRT_SECURE_NO_WARNINGS
#include <direct.h>
#elif defined(__unix)
#include <unistd.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif
#if defined(_OPENMP)
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

  m_nthreads = (*this)("threads", 1);

#if defined(_OPENMP)
  const char * omp_var = std::getenv("OMP_NUM_THREADS");
  if (omp_var) m_nthreads = strtol(omp_var, NULL, 10);
  omp_set_num_threads(m_nthreads);
#else
  m_nthreads = 1;
#endif

  m_diffusion = (*this)("diffusion", true);
  m_convection = (*this)("convection", true);
  m_n_global[0] = (*this)("n", 400);
  m_n_global[1] = (*this)("m", 400);
  m_n_global[2] = (*this)("p", 400);
  m_itmax = (*this)("it", 10);
  double dt_max = 1.5/(m_n_global[0]*m_n_global[0]
		       + m_n_global[1]*m_n_global[1]
		       + m_n_global[2]*m_n_global[2]);
  m_dt = (*this)("dt", dt_max);
  m_freq = (*this)("out", -1);

  m_convection = (*this)("convection", 1) == 1;
  m_diffusion = (*this)("diffusion", 1) == 1;
  
  if (!m_help) {

    int p[3];
    
    if (m_dt > dt_max)
      std::cerr << "Warning : provided dt (" << m_dt
		<< ") is greater then the recommended maximum (" <<  dt_max
		<< ")" << std::endl;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &m_size);
    int periods[3], coords[3];

    int i;
    for (i=0; i<3; i++) {
      p[i] = (m_n_global[i] > 1) ? 0 : 1;
      periods[i] = 0;
    }

    // Creation d'une "grille" de processus MPI
    MPI_Dims_create(m_size, 3, p);

    MPI_Cart_create(MPI_COMM_WORLD, 3, p, periods, 1, &m_comm);
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Cart_coords(m_comm, m_rank, 3, coords);

    for (i=0; i<3; i++) {
      m_neigh[i][0] = MPI_PROC_NULL;
      m_neigh[i][1] = MPI_PROC_NULL;
      MPI_Cart_shift(m_comm, i, 1, &m_neigh[i][0], &m_neigh[i][1]);
    }

    for (i=0; i<3; i++) {
      m_dx[i] = m_n_global[i]>1 ? 1.0/(m_n_global[i]-1) : 0.0;
      m_n[i] = (m_n_global[i]-2)/p[i]+2;
      
      m_imin_global[i] = (m_n[i]-2) * coords[i] + 1;
      m_xmin[i] = m_dx[i] * (m_imin_global[i]-1);

      if (coords[i] < p[i]-1)
        m_imax_global[i] = m_imin_global[i] + m_n[i] - 2;
      else {
        m_imax_global[i] = m_n_global[i]-1;
        m_n[i] = m_imax_global[i] - m_imin_global[i] + 2;
      }
      
      m_di[i] = 1;
      if (m_n[i] < 2) {
       	m_imax_global[i] = m_imin_global[i] + 1; m_di[i] = 0;
      }
      m_imin[i] = 1;
      m_imax[i] = m_n[i] - 1;
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
  if (!m_help)
    MPI_Finalize();

  if (m_out)
    delete m_out;
}

std::ostream & operator<<(std::ostream &f, const Parameters & p)
{
  f << "Process : " << p.rank()+1 << "/" << p.size() << "\n";

  if (p.rank() == 0) 
    f << "Whole domain :   " << std::endl << "   "
      << "[" << 0 << "," << p.n_global(0) - 1  << "] x "
      << "[" << 0 << "," << p.n_global(1) - 1  << "] x "
      << "[" << 0 << "," << p.n_global(2) - 1  << "]"
      << "\n\n";

  f << "Domain (interior points) :   " << std::endl << "   "
    << "[" << p.imin_global(0) << "," << p.imax_global(0)-1  << "] x "
    << "[" << p.imin_global(1) << "," << p.imax_global(1)-1 << "] x "
    << "[" << p.imin_global(2) << "," << p.imax_global(2)-1  << "]"
    << "\n\n";

  f << p.nthreads() << " thread(s)\n"
    << "It. max : " << p.itmax() << "\n"
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
