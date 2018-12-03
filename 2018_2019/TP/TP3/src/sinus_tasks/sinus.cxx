#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#define NTHREADS omp_get_num_threads()
#define ITHREAD  omp_get_thread_num()
#else
#define NTHREADS 1
#define ITHREAD  0
#endif

#include "sin.hxx"

void init(std::vector<double> & pos,
          std::vector<double> & v1,
          std::vector<double> & v2,
          int n1, int n2)
{
  double x, pi = 3.14159265;
  int i, n = pos.size();
  
  for (i=n1; i<n2; i++) {
    x = i*2*pi/n;
    pos[i] = x;
    v1[i] = sinus_machine(x);
    v2[i] = sinus_taylor(x);
  }
}

void save(const char *filename,
	  std::vector<double> & pos,
	  std::vector<double> & v1,
	  std::vector<double> & v2)
{
  std::ofstream f(filename);

  f  << "# x sin(systeme) approximation" << std::endl;
  int i, n = pos.size();
  for (i=0; i<n; i++)
    f << pos[i] << " " << v1[i] << " " << v2[i] << std::endl;
}

void print_times(const std::vector<int> & binds,
                 const std::vector<double> & start,
                 const std::vector<double> & end,
                 double tref)
{
  int i, n = binds.size();
  std::vector<double> x1(n), x2(n), y1(n), y2(n);
  double ymin = 1e+20, ymax = -1e+20;
  double xmin = 1e+20, xmax = -1e+20;

  for (i=0; i<n; i++) {
    y1[i] = start[i] - tref;
    if (y1[i] < ymin) ymin =  y1[i];
    y2[i] = end[i] - tref;
    if (y2[i] > ymax) ymax =  y2[i];

    x1[i] = binds[i] - 0.4;
    x2[i] = binds[i] + 0.4;
    
    if (x1[i] < xmin)
      xmin = x1[i];
    else if (x2[i] > xmax)
      xmax = x2[i];
  }
  xmin -= 0.5;
  xmax += 0.5;
  double dy = ymax - ymin;
  ymax += dy*0.1;
  ymin -= dy*0.1;
  
  std::ofstream f("r_times");

  f << "set terminal pdf\n"
    << "set output 'r_times.pdf'\n";

  f << "set xrange[" << xmin << ":" << xmax << "]\n" 
    << "set yrange[" << ymin << ":" << ymax << "]\n";
  
  for (i=0; i<n; i++) {

    
    f << "set object " << i+1 << " rect "
      << " from " << x1[i] << "," << y1[i]
      << " to   " << x2[i] << "," << y2[i]
      << std::endl;
    f << "set label " << i+1 << "'Task " << i << "' at "
      << 0.5*(x1[i]+x2[i]) << "," << 0.5*(y1[i]+y2[i]) << " center"
      << std::endl;
      }
  
  f << "set xlabel 'Thread'\n set ylabel 'Time (s)'\n"
    << "set xtics 1\n"
    << "unset key\nplot -1" << std::endl;
    
}

void stat(const std::vector<double> & v1,
          const std::vector<double> & v2,
          double & sum1, double & sum2,
          int n1, int n2)
{
  double s1 = 0.0, s2 = 0.0, err;
  int i;
  for (i=n1; i<n2; i++) {
    err = v1[i] - v2[i];
    s1 += err;
    s2 += err*err;
  }

  sum1 += s1;
  sum2 += s2;
}

int main(int argc, char **argv)
{
  double t0 = omp_get_wtime();
  int nthreads;
  #pragma omp parallel
  {
    #pragma omp master
    nthreads = NTHREADS;
  }

  size_t n = argc > 1 ? strtol(argv[1], nullptr, 10) : 2000;
  int imax = argc > 2 ? strtol(argv[2], nullptr, 10) : IMAX;
  set_terms(imax);

  int ntasks = 5*nthreads;
  std::cout << "\n\nversion OpenMP tasks : \n\t" << nthreads << " thread(s)"
            << "  " << ntasks << " tasks\n"
            << "\ttaille vecteur = " << n << "\n"
            << "\ttermes (formule Taylor) : " << imax
            << std::endl;

  std::vector<double> elapsed_init(ntasks), elapsed_stat(ntasks);
  std::vector<double> task_start_init(ntasks), task_end_init(ntasks);
  std::vector<int> thread_task_init(ntasks), thread_task_stat(ntasks);
  
  std::vector<double> pos(n), v1(n), v2(n);
  double m, e;

  int dn = n/ntasks;

#pragma omp parallel
  {
#pragma omp master
    {
      for (int i=0; i<ntasks-1; i++)
#pragma omp task firstprivate(i) 
        {
          task_start_init[i] = omp_get_wtime();
          
          int n_start = i * dn;
          int n_end = (i+1) * dn;
          init(pos, v1, v2, n_start, n_end);
          
          task_end_init[i] = omp_get_wtime();
          elapsed_init[i] += task_end_init[i] - task_start_init[i];
          thread_task_init[i] = ITHREAD;
        }
#pragma omp task
      {
        int i = ntasks-1;
        task_start_init[i] = omp_get_wtime();

        int n_start = i*dn;
        int n_end = n;
        init(pos, v1, v2, n_start, n_end);
        
        task_end_init[i] = omp_get_wtime();
        elapsed_init[i] += task_end_init[i] - task_start_init[i];
        thread_task_init[i] = ITHREAD;
      }
#pragma omp taskwait
    
      if (n < 10000)
        save("sinus.dat", pos, v1, v2);
    
      // t0 = omp_get_wtime();
      for (int i=0; i<ntasks-1; i++)
#pragma omp task firstprivate(i) 
        {
          double t0 = omp_get_wtime();
          
          int n_start = i * dn;
          int n_end = (i+1) * dn;
          stat(v1, v2, m, e, n_start, n_end);
          
          elapsed_stat[i] += omp_get_wtime() - t0;
          thread_task_stat[i] = ITHREAD;
        }
#pragma omp task
      {
        double t0 = omp_get_wtime();
        
        int n_start = (ntasks-1)*dn;
        int n_end = n;
        stat(v1, v2, m, e, n_start, n_end);
        
        elapsed_stat[ntasks-1] += omp_get_wtime() - t0;
        thread_task_stat[ntasks-1] = ITHREAD;
      }
#pragma omp taskwait
      m /= n;
      e = sqrt(e/n - m*m);
      // elapsed_stat = omp_get_wtime() - t0;
      
    }
  }
  std::cout << "erreur moyenne : " << m << " ecart-type : " << e
            << std::endl << std::endl;
  
  for (int i=0; i<ntasks; i++)
    std::cout << "time (task " << std::setw(3) << i << ") :"
              << " init (thread " << thread_task_init[i] << ") "
              << std::setw(9) << elapsed_init[i] << "s "
              << " stat (thread " << thread_task_stat[i] << ") "
              << std::setw(9) << elapsed_stat[i] << std::endl;

  std::cout << "time :" << omp_get_wtime() - t0 << std::endl;

  print_times(thread_task_init, task_start_init, task_end_init, t0);
  return 0;
}
