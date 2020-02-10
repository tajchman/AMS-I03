#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>

#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#define NTHREADS omp_get_num_threads()
#define ITHREAD  omp_get_thread_num()
#else
#define NTHREADS 1
#define ITHREAD  0
#endif

#include "sin.hxx"
#include "timer.hxx"

void init(std::vector<double> & pos,
          std::vector<double> & v1,
          std::vector<double> & v2,
          int n1, int n2, int m1, int m2, int nTotal)
{
  double x, pi = 3.14159265;
  int i;

  for (i=n1; i<n2; i++) {
    x = (i+m1)*2*pi/nTotal;
    pos[i] = x;
    v1[i] = sinus_machine(x);
    v2[i] = sinus_taylor(x);
  }
}

void save(const char *filename,
	  std::vector<double> & pos,
	  std::vector<double> & v1,
	  std::vector<double> & v2,
          int iproc, int nprocs)
{
  std::ofstream f(filename);

  f  << "# x sin(systeme) approximation" << std::endl;
  int i, n = pos.size();
  for (i=0; i<n; i++)
    f << pos[i] << " " << v1[i] << " " << v2[i] << std::endl;

  if (iproc == 0) {
    std::ofstream t("sinus_mpi.gnp");
    t << "set output 'sinus_mpi.pdf'\n"
      << "set term pdf\n"
      << "plot ";
    for (int i=0; i<nprocs; i++) {
      if (i > 0) t << ", ";
      t << "'sinus_" << i << ".dat' using 1:2 notitle w l lw 3, 'sinus_" << i << ".dat' using ($1):($3+0.03) notitle w l lw 3";
    }
    t << std::endl;
  }
}

void stat(const std::vector<double> & v1,
          const std::vector<double> & v2,
          int n1, int n2,
          double & sum1, double & sum2)
{
  double s1 = 0.0, s2 = 0.0, err;
  int i;

  for (i=n1; i<n2; i++) {
    err = v1[i] - v2[i];
    s1 += err;
    s2 += err*err;
  }

  sum1 = s1;
  sum2 = s2;
}

int main(int argc, char **argv)
{
  Timer T_total;
  T_total.start();
  
  int nthreads;
  #pragma omp parallel
  {
    #pragma omp master
    nthreads = NTHREADS;
  }

  size_t n = argc > 1 ? strtol(argv[1], nullptr, 10) : 2000;
  int imax = argc > 2 ? strtol(argv[2], nullptr, 10) : IMAX;
  set_terms(imax);
  
  int nprocs, iproc;
  
  int required = MPI_THREAD_FUNNELED, provided;
  MPI_Init_thread(&argc, &argv, required, &provided);
  if (provided < required) {
     std::cerr << "Interaction MPI - OpenMP insuffisante" << std::endl;
     MPI_Finalize();
     return -1;
  }
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &iproc);

  int dn_MPI = n/nprocs;
  int n1_MPI = dn_MPI * iproc;
  int n2_MPI = (iproc < nprocs-1) ? n1_MPI + dn_MPI : n;
  dn_MPI = n2_MPI - n1_MPI;

  set_terms(imax);

  if (iproc == 0)
    std::cout << "\n\nversion mpi - openmp (coarse grain) : \n"
              << "\t" << nprocs << " processus MPI - " << nthreads << " threads/processus\n"
              << "\ttaille vecteur = " << n << "\n"
              << "\ttermes (formule Taylor) : " << imax << "\n";
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "\tzone locale "
            << iproc << " : (" << n1_MPI << ", " << n2_MPI << ")\n";

  std::vector<int> n_start(nthreads), n_end(nthreads);  
  int dn_openmp;
  
  dn_openmp = dn_MPI/nthreads;
  for (int i=0; i<nthreads-1; i++) {
    n_start[i] = i * dn_openmp;
    n_end[i] = (i+1) * dn_openmp;
  }
  n_start[nthreads-1] = (nthreads-1)*dn_openmp;
  n_end[nthreads-1] = dn_MPI;
     
  std::vector<double> pos(dn_MPI), v1(pos.size()), v2(pos.size());
  double m, e, m_local, e_local;

  m = 0;
  e = 0;

  std::vector<double> elapsed_init(nthreads), elapsed_stat(nthreads);
  
#pragma omp parallel shared(pos, v1, v2, n) reduction(+: e_local, m_local)
  {
    Timer t_init, t_stat;
    int ithread = ITHREAD;
    int n1 = n_start[ithread], n2 = n_end[ithread];
    
    t_init.start();
    init(pos, v1, v2, n1, n2, n1_MPI, n2_MPI, n);
    t_init.stop();
    elapsed_init[ithread] =  t_init.elapsed();

    MPI_Barrier(MPI_COMM_WORLD);

    #pragma omp single
    {
      if (n < 10000) {
        std::string s = "sinus_";
        s += std::to_string(iproc) + ".dat";
        save(s.c_str(), pos, v1, v2, iproc, nprocs);
        }
    }
  
    t_stat.start();
    stat(v1, v2, n1, n2, m_local, e_local);
    t_stat.stop();
    elapsed_stat[ithread] =  t_stat.elapsed();
  }

  MPI_Allreduce(&m_local, &m, 1, MPI_DOUBLE, MPI_SUM,
             MPI_COMM_WORLD);
  MPI_Allreduce(&e_local, &e, 1, MPI_DOUBLE, MPI_SUM,
             MPI_COMM_WORLD);
  m = m/n;
  e = sqrt(e/n - m*m);

  if (iproc == 0) {
    std::cout << "erreur moyenne : " << m << " ecart-type : " << e
              << std::endl << std::endl;
  }
  
  MPI_Finalize();

  for (int i=0; i<nthreads; i++) {
    std::cout << "time init (rank " << iproc << ", thread " << i << ") : "
	      << std::setw(12) << elapsed_init[i] << " s" << std::endl; 
    std::cout << "time stat (rank " << iproc << ", thread " << i << ") : "
	      << std::setw(12) << elapsed_stat[i] << " s" << std::endl;
  }
  T_total.stop();
  if (iproc == 0)
    std::cout << "time : "
            << std::setw(12) << T_total.elapsed() << " s" << std::endl;  
  return 0;
}
