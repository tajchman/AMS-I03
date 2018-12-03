#include <iostream>
#include <iomanip>
#include <fstream>
#include "charge.hxx"

Charge::Charge(int n, int nthreads)
  : m_nthreads(nthreads), m_n(n),
    m_elapsed(nthreads), m_modified(false), m_bornes(nthreads) {

  int i;
  
  m_fileName = std::string("loads_") + std::to_string(n)
    + "_" + std::to_string(nthreads);
  
  std::ifstream f(m_fileName.c_str());
  if (f) {
    for (i=0; i<m_nthreads; i++)
      f >> m_bornes[i].first >> m_bornes[i].second;
  }
  else {
    int dn = m_n/m_nthreads;
    for (int i=0; i<m_nthreads-1; i++) {
      m_bornes[i].first  = i * dn;
      m_bornes[i].second = (i+1) * dn;
    }
    m_bornes[m_nthreads-1].first = (m_nthreads-1)*dn;
    m_bornes[m_nthreads-1].second = m_n;
  }
}

void Charge::update(const std::vector<double> elapsed)
{
  std::vector<double> t(m_nthreads + 1), pos(m_nthreads);
  double t_moyen, tt;
  int i, j;

  t[0] = 0.0;
  for (i=1; i<=m_nthreads; i++)
    t[i] = elapsed[i-1] + t[i-1];
  
  t_moyen = t[m_nthreads]/m_nthreads;
    
  j = 0;
  for (i=1; i<m_nthreads; i++) {

    tt = i*t_moyen;
    while (t[j+1] < tt) j++;
    
    int m = (tt - t[j])/(t[j+1] - t[j])
      * (m_bornes[j].second - m_bornes[j].first);
    pos[i] = m_bornes[j].first + m;
    
  }

  for (i=1; i<m_nthreads; i++)
    m_bornes[i-1].second = m_bornes[i].first = pos[i];
  
  m_modified = true;
}

Charge::~Charge()
{
  int i;
  if (m_modified) {
    std::ofstream f(m_fileName.c_str());
    if (f) {
      for (i=0; i<m_nthreads; i++)
        f << m_bornes[i].first << " " << m_bornes[i].second << std::endl;
    }
  }
}

