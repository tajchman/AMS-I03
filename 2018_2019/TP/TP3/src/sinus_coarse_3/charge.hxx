#ifndef _CHARGE_HXX
#define _CHARGE_HXX

#include <vector>

class Charge {
public :
  Charge(int n, int nthreads);
  ~Charge();

  void update(const std::vector<double> elapsed);
  
  int min(int i) const { return m_bornes[i].first; }
  int max(int i) const { return m_bornes[i].second; }
  
private:
  bool m_modified;
  std::vector<double> m_elapsed;
  std::vector<std::pair<int, int>> m_bornes;
  int m_n, m_nthreads;
  std::string m_fileName;
};
  

#endif
