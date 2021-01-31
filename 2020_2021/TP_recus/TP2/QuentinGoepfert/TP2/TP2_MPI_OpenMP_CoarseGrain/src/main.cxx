#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "parameters.hxx"
#include "values.hxx"
#include "scheme.hxx"
#include "timer.hxx"
#include "os.hxx"

double cond_ini(const std::array<double, 3> & x)
{
  double xc = x[0] - 0.5;
  double yc = x[1] - 0.5;
  double zc = x[2] - 0.5;

  if (xc*xc+yc*yc+zc*zc < 0.09)
    return 1.0;
  else
    return 0.0;
}

double cond_lim(const std::array<double, 3> & x)
{
  return 2.0;
}

double force(const std::array<double, 3> & x)
{
  if (x[0] < 0.3)
    return 0.0;
  else
    return sin(x[0]-0.5) * cos(x[1]-0.5) * exp(- x[2]*x[2]);
}

int main(int argc, char *argv[])
{
  AddTimer("total");
  AddTimer("init");
  AddTimer("calcul");
  AddTimer("comm");
  AddTimer("other");

  Timer & T_total = GetTimer(0);
  Timer & T_init = GetTimer(1);
  Timer & T_calcul = GetTimer(2);
  Timer & T_comm = GetTimer(3);
  Timer & T_other = GetTimer(4);

  T_total.start();
  int provided;
 
  MPI_Init_thread( 0, 0, MPI_THREAD_MULTIPLE, &provided );
  if (MPI_THREAD_MULTIPLE != provided)
  {
    printf("Could not give the requested MPI access to threads");
    return -1 ;
  }

  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Parameters Prm(argc, argv, size, rank);
  if (Prm.help()) return 0;

  std::cout << Prm << std::endl;

  int itMax = Prm.itmax();
  int freq = Prm.freq();

  T_init.start();

  Scheme C(Prm, force);

  Values u_0(Prm);
  u_0.boundaries(cond_lim);
  u_0.init(cond_ini);
  C.setInput(u_0);
  T_init.stop();

  MPI_Barrier(Prm.comm());
  if (Prm.rank() == 0) {
    std::cout << "\n  temps init "
              << std::setw(10) << std::setprecision(6)
              << T_init.elapsed() << " s\n" << std::endl;
  }

  # pragma omp parallel default ( shared )
  {
    for (int it=0; it < itMax; it++) {
      
      # pragma omp master
      {
        if (freq > 0 && it % freq == 0) {
          T_other.start();
          C.getOutput().plot(it);
          T_other.stop();
        }
      
        T_comm.start();
        C.synchronize();
        T_comm.stop();
      }
      #pragma omp barrier


      #pragma omp master
      T_calcul.start();
      C.iteration();
      
      #pragma omp barrier

      #pragma omp master
      {
        T_calcul.stop();

        if (Prm.rank() == 0) {
          std::cout << "iter. " << std::setw(3) << it
          << "  variation " << std::setw(10) << std::setprecision(4) << C.variation()
          << "  temps calcul " << std::setw(8) << std::setprecision(3)
          << T_calcul.elapsed() << " s"
          << "  comm. " << std::setw(8) << std::setprecision(3)
          << T_comm.elapsed() << " s"
          << std::endl;
        } 
      }

    #pragma omp barrier
    }
  }

  if (freq > 0 && itMax % freq == 0) {
    T_other.start();
    C.getOutput().plot(itMax);
    T_other.stop();
  }

  MPI_Finalize();

  T_total.stop();

  if (Prm.rank() == 0)
    std::cout << "\n" << std::setw(26) << "temps total"
              << std::setw(10) << T_total.elapsed() << " s\n" << std::endl;

  #ifdef _OPENMP
    int id = Prm.nthreads();
  #else
    int id = 0;
  #endif

  if (Prm.rank() == 0) {
    std::string s = Prm.resultPath();
    mkdir_p(s.c_str());
    s += "/temps_t_";
    s += std::to_string(id);
    s += "_p_";
    s += std::to_string(Prm.size());
    s += ".dat";
    std::ofstream f(s.c_str());
    f << id << " " << Prm.size() << " " 
      << T_total.elapsed() << " " << C.variation() << std::endl;
  }

  return 0;
}
