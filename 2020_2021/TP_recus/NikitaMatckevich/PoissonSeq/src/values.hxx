#ifndef __VALUES__
#define __VALUES__

#include "parameters.hxx"
#include <vector>
#include <iostream>

class Values {
public:
  Values(const Parameters* p);

  Values(const Values&) = delete; // same as private constructor but allows
                                  // better operator=
  Values(Values&&); // move semantics for faster initialization? u0 never used
  Values& operator=(Values) noexcept; // noexcept operator=
  void swap(Values& other) noexcept; // noexcept swap with move semantics?

  virtual ~Values() = default; // inherit something from vectors of data ??

  void init();
  void init(const callback_t& f); //why copying std::function here?
  void boundaries(const callback_t& f);

  inline double& operator() (int i, int j, int k) {
    return m_u[n2()*i + n1()*j + k];
  }
  inline double operator() (int i, int j, int k) const {
    return m_u[n2()*i + n1()*j + k];
  }

  inline int size(int i) const { return m_n(i); }
  void print(std::ostream& f) const;
  void plot(int order) const;

private:

  inline int m_n(int i) const noexcept { return m_p->n(i); }
  inline int n1() const noexcept { return m_n(2); };
  inline int n2() const noexcept { return m_n(2)*m_n(1); }

  inline int imin() const noexcept { return m_p->imin(0); }
  inline int jmin() const noexcept { return m_p->imin(1); }
  inline int kmin() const noexcept { return m_p->imin(2); }

  inline int imax() const noexcept { return m_p->imax(0); }
  inline int jmax() const noexcept { return m_p->imax(1); }
  inline int kmax() const noexcept { return m_p->imax(2); }

  inline double dx() const noexcept { return m_p->dx(0); }
  inline double dy() const noexcept { return m_p->dx(1); }
  inline double dz() const noexcept { return m_p->dx(2); }
  
  inline double xmin() const noexcept { return m_p->xmin(0); }
  inline double ymin() const noexcept { return m_p->xmin(0); }
  inline double zmin() const noexcept { return m_p->xmin(0); }

  inline double xmax() const noexcept { return xmin() + (imax()-imin()) * dx(); }
  inline double ymax() const noexcept { return ymin() + (jmax()-jmin()) * dy(); }
  inline double zmax() const noexcept { return zmin() + (kmax()-kmin()) * dz(); }

  const Parameters* m_p;
  std::vector<double> m_u;  
};
				   
std::ostream& operator<< (std::ostream& f, const Values& v);

#endif
