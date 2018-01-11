#ifndef __PARAMETERS__
#define __PARAMETERS__

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "GetPot.hxx"
#include <mpi.h>

class Parameters : public GetPot {
public:

  Parameters(int *, char ***);
  ~Parameters();
  std::ostream & out() { return *m_out; }
  void info();

  MPI_Comm comm() const { return m_comm; }

  int size() const { return m_size; }
  int rank() const { return m_rank; }
  int neighbor(int idim, int j) const { return m_neigh[idim][j]; }
  
  const int *n() const { return m_n; }
  const double *dx() const { return m_dx; }
  const double *xmin() const { return m_xmin; }

  int p(int i) const { return m_p[i]; }
  int n(int i) const { return m_n[i]; }
  int nmax(int i) const { return m_nmax[i]; }
  int p0(int i) const { return m_p0[i]; }
  double dx(int i) const { return m_dx[i]; }
  double xmin(int i) const { return m_xmin[i]; }

  int imin(int i) const { return m_imin[i]; }
  int imax(int i) const { return m_imax[i]; }
  int di(int i) const { return m_di[i]; }
  
  int itmax() const { return m_itmax; }
  double dt() const { return m_dt; }

  int freq() const { return m_freq; }
  std::string resultPath() const { return m_path; }
  bool help();
  
  bool convection() const { return m_convection; }
  bool diffusion() const { return m_diffusion; }

  void convection(bool b) { m_convection = b; }
  void diffusion(bool b) { m_diffusion = b; }

private:

  std::string m_command;
  std::ostream * m_out;
  int m_rank, m_size, m_p[3];

  MPI_Comm m_comm;

  int m_neigh[3][2];
  int m_p0[3], m_n[3], m_nmax[3];
  double m_dx[3], m_xmin[3];
  int m_imin[3], m_imax[3], m_di[3];
  
  int m_itmax;
  double m_dt;
  bool m_convection, m_diffusion;

  int m_freq;

  std::string m_path;
  bool m_help;

};

std::ostream & operator<<(std::ostream &f, const Parameters & p);


#endif
