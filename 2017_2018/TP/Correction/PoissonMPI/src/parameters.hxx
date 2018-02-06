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

  Parameters(int argc, char **argv);
  ~Parameters();
  std::ostream & out();
  void info();

  MPI_Comm comm() const { return m_comm; }

  int size() const { return m_size; }
  int rank() const { return m_rank; }
  int neighbor(int idim, int j) const { return m_neigh[idim][j]; }
  
  int nmax(int i) const { return m_nmax[i]; }
 
  int n(int i) const { return m_n[i]; }
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
  std::ostream * m_out;

  std::string m_command;
  int m_rank, m_size;

  MPI_Comm m_comm;

  int m_neigh[3][2];
  int m_n[3], m_nmax[3];
  double m_xmin[3], m_dx[3];
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
