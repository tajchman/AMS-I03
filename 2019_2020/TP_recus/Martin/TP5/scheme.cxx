#include "scheme.hxx"
#include "parameters.hxx"

#ifdef _OPENMP
#include <omp.h>
#define NTHREADS omp_get_num_threads()
#define ITHREAD  omp_get_thread_num()
#else
#define NTHREADS 1
#define ITHREAD  0
#endif

#include <mpi.h>
#include <sstream>
#include <iomanip>


void Scheme::initialize(const Parameters *P)
{
  m_P = P;
  m_u.init(P);
  m_v.init(P);
  int i;
  for (i=0; i<3; i++) {
    m_n[i] = m_P->n(i);
    m_dx[i] = m_P->dx(i);
    m_di[i] = (m_n[i] < 2) ? 0 : 1;
  }

  kStep = 1;
  m_t = 0.0;

  m_duv = 0.0;
}

Scheme::~Scheme()
{
}

double Scheme::present()
{
  return m_t;
}

size_t Scheme::getDomainSize(int dim) const
{
  size_t d;
  switch (dim) {
    case 0:
      d = m_n[0];
      break;
    case 1:
      d = m_n[1];
      break;
    case 2:
      d = m_n[2];
      break;
    default:
      d = 1;
  }
  return d;
}


bool Scheme::solve(unsigned int nSteps)
{
  m_timers[1].start();

  double du_sum, du;
  double dx2 = m_dx[0]*m_dx[0] + m_dx[1]*m_dx[1] + m_dx[2]*m_dx[2];
  double dt = 0.5*(dx2 + 1e-12);
  double lambda = 0.25*dt/(dx2 + 1e-12);

  int i, j, k;
  size_t iStep;
  int   di = m_di[0],     dj = m_di[1],     dk = m_di[2];
  int imin = 1, imax = m_n[0] - 1;
  int jmin = 1, jmax = m_n[1] - 1;
  int kmin = 1, kmax = m_n[2] - 1;
  
  
  //on verifie que ne nombre de threads donnés est bien suffisante
  int required = MPI_THREAD_FUNNELED, provided;
  MPI_Init_thread(&argc, &argv, required, &provided);
  if (provided < required) {
     std::cerr << "Interaction MPI - OpenMP insuffisante" << std::endl;
     MPI_Finalize();
     return -1;
  }

  for (iStep=0; iStep < nSteps; iStep++) {//bucle en temps
#pragma omp parallel sections
{
	#pragma omp section
	{
    m_timers[1].start();

    du_sum = 0.0;

    for (i = imin; i < imax; i++)  //boucle sans besoin d'echanges, sont les données à l'interieur
      for (j = jmin; j < jmax; j++)  //il faudra mettre ces bucles dans un thread
        for (k = kmin; k < kmax; k++) {
          du = 6 * m_u(i, j, k) 
              - m_u(i + di, j, k) - m_u(i - di, j, k)
              - m_u(i, j + dj, k) - m_u(i, j - dj, k) 
              - m_u(i, j, k + dk) - m_u(i, j, k - dk);
          du *= lambda;
          m_v(i, j, k) = m_u(i, j, k) - du;
          du_sum += du > 0 ? du : -du;
        }

    m_timers[1].stop();
	}
	#pragma omp section
	{
    m_timers[2].start();

    double du_sum_global;
    MPI_Allreduce(&du_sum, &du_sum_global, 1, MPI_DOUBLE, MPI_SUM, m_P->comm());
    du_sum = du_sum_global;

    m_v.synchronize();  //calculs sur le bord du processus. Alors ici est ou les communications et les calculs avec ces donnés sont faites
    					//De cet façon on doit separer cet partie des trois boucles for de la lgne 78
    					//on met m_v.synchronize() dans un autre thread

    m_timers[2].stop();
	}
}
    m_timers[1].start();

    m_u.swap(m_v);
    m_t += dt;

    m_timers[1].stop();
    m_timers[3].start();
    if (m_P->rank() == 0) {
      std::cerr << " iteration " << std::setw(4) << kStep
              << " variation " << std::setw(12) << std::setprecision(6) << du_sum;
      size_t i, n = m_timers.size();
      std::cerr << " (times :";
      for (i=0; i<n; i++)
	std::cerr << " " << std::setw(5) << m_timers[i].name()
	          << " " << std::setw(9) << std::fixed << m_timers[i].elapsed();
      std::cerr	  << ")   \n";
    }
    m_timers[3].stop();

    kStep++;
  }

  m_duv = du_sum;

  return true;
}

double Scheme::variation()
{
  return m_duv;
}

void Scheme::terminate() {
  if (m_P->rank() == 0)
    std::cerr << "\n\nterminate " << codeName << std::endl;
}

const Values & Scheme::getOutput()
{
  return m_u;
}

void Scheme::setInput(const Values & u)
{
  m_u = u;
  m_v = u;
}

void Scheme::save(const char * /*fName*/)
{
}


