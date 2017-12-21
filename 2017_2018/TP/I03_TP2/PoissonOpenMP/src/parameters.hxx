#ifndef __PARAMETERS__
#define __PARAMETERS__

#include <iostream>
#include <string>

#include "GetPot.hxx"

class Parameters : public GetPot {
public:

  Parameters(int argc, char **argv);
  ~Parameters();

  bool contains(const char *varname);
  friend std::ostream & operator << (std::ostream &f, const Parameters & P);
  std::ostream & out() { return *m_out; }
  void info();

  int n(int i) const { return m_n[i]; }
  double dx(int i) const { return m_dx[i]; }

  int imin(int i) const { return m_imin[i]; }
  int imax(int i) const { return m_imax[i]; }
  int di(int i) const { return m_di[i]; }
  
  int itmax() const { return m_itmax; }
  double dt() const { return m_dt; }
  int nthreads() const { return m_nthreads; };

  int output() const { return m_output; }
  std::string resultPath() { return m_path; }
  bool help();

  bool convection()  const{ return m_convection; }
  bool diffusion() const { return m_diffusion; }

  void convection(bool b) { m_convection = b; }
  void diffusion(bool b) { m_diffusion = b; }
  
private:
  std::ostream * m_out;

  int m_n[3];
  double m_dx[3];
  int m_imin[3], m_imax[3], m_di[3];
  
  int m_itmax;
  double m_dt;
  bool m_convection, m_diffusion;
  int m_nthreads;
  
  int m_output;

  std::string m_path;
  bool m_help;

};

std::ostream & operator << (std::ostream &f, const Parameters & P);

#endif
