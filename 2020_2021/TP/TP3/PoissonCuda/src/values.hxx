#ifndef __VALUES__
#define __VALUES__

#include "parameters.hxx"
#include <vector>
#include <iostream>

class Values {

public:

  Values(Parameters & p);
  virtual ~Values();
  void operator= (const Values &);
  
  void init();
  void zero();
  void boundaries();

  double & operator() (int i,int j,int k) {
    return h_u[n2*k + n1*j + i];
  }
  double operator() (int i,int j,int k) const {
    return h_u[n2*k + n1*j + i];
  }
  
  void plot(int order) const;
  void swap(Values & other);
  int size(int i) const { return m_imax[i] - m_imin[i] + 1; }
  void print(std::ostream &f) const;

  double * dataCPU() { return h_u; }
  double * dataGPU() { return d_u; }
  void synchronized(bool b) { h_synchronized = b; }
  
private:

  Values(const Values &) = delete;
  int n1, n2, nn;
  double * d_u, * h_u;
  mutable bool h_synchronized;

  Parameters & m_p;
  int m_imin[3];
  int m_imax[3];
  int m_n_local[3];

  double m_dx[3];
  double m_xmin[3];
  double m_xmax[3];

};

std::ostream & operator<< (std::ostream & f, const Values & v);

void zeroWrapper(double *d, int n);
void initWrapper(double *d, int n[3]);
void boundariesWrapper(double *d, int n[3], int imin[3], int imax[3]);

double * allocate(int n);
void deallocate(double *&d);
void copyDeviceToHost(double *h, double *d, int n);
void copyHostToDevice(double *d, double *h, int n);
void copyDeviceToDevice(double *d_out, double *d_in, int n);

#endif
