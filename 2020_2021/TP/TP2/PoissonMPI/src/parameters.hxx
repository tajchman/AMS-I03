#ifndef __PARAMETERS__
#define __PARAMETERS__

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <mpi.h>

#include "arguments.hxx"

typedef std::function<double(double, double, double)> callback_t;

class Parameters : public Arguments {
public:

  Parameters(int argc, char **argv, int size, int rank);
  void info();

  int n(int i) const { return m_n[i]; }
  double dx(int i) const { return m_dx[i]; }
  double xmin(int i) const { return m_xmin[i]; }
  double xmax(int i) const { return m_xmin[i]; }

  int imin(int i) const { return m_imin[i]; }
  int imax(int i) const { return m_imax[i]; }
  
  int itmax() const { return m_itmax; }
  double dt() const { return m_dt; }

  int freq() const { return m_freq; }
  std::string resultPath() const { return m_path; }
  bool help();

  int rank() const { return m_rank; }
  int size() const { return m_size; }

private:

  std::string m_command;
  int m_n[3], m_n_global[3];
  double m_xmin[3], m_xmax[3], m_dx[3];
  int m_imin[3], m_imax[3];
  
  int m_itmax;
  double m_dt;
  
  int m_freq;

  std::string m_path;
  bool m_help;

  int m_size, m_rank;
  MPI_Comm m_comm;
};

std::ostream & operator<<(std::ostream &f, const Parameters & p);


#endif
