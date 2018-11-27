#include <fstream>
#include "charge.hxx"

Charge::Charge(int n, int nthreads)
  : m_nthreads(nthreads), m_n(n),
    m_elapsed(nthreads), m_modified(false), m_bornes(nthreads) {

  int i;
  
  m_fileName = std::string("times_") + std::to_string(n)
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
  std::vector<double> t = elapsed;
  double t_moyen = 0.0;
  int i;
  
  for (i=0; i<m_nthreads; i++)
    t_moyen += t[i];
  t_moyen /= m_nthreads;

  for (i=0; i<m_nthreads-1; i++) {
    double tp = t[i+1]/(m_bornes[i].second - m_bornes[i].first);
    
    if (t[i] < t_moyen) {
      int dn = (t_moyen - t[i])/tp;
      m_bornes[i].second += dn;
      m_bornes[i+1].first = m_bornes[i].second;
      t[i+1] -= dn * tp;
    }
    else if (t[i] > t_moyen) {
      int dn = (t[i] - t_moyen)/tp;
      m_bornes[i].second -= dn;
      m_bornes[i+1].first = m_bornes[i].second;
      t[i+1] += dn * tp;
    }
  }
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

