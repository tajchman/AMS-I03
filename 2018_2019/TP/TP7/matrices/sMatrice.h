#include <string>
#include <iostream>
#include <iomanip>
#include <vector>

class sMatrice {
public:
 sMatrice(int n, int m, const char *s)
   : _n(n), _m(m), _coefs(n*m) {}

  double & operator()(int i, int j) { return _coefs[i*_m+j]; }
  double operator()(int i, int j) const { return _coefs[i*_m+j]; }

  int n() const { return _n; }
  int m() const { return _m; }
  double * data() { return _coefs.data(); }
  const double * data() const { return _coefs.data(); }
  const char * name() const { return _name.c_str(); }
  
private:
  int _n, _m;
  std::vector<double> _coefs;
  std::string _name;
};

inline std::ostream & operator<<(std::ostream & f, const sMatrice & M)
{
  int i,j;
  f << M.name() << std::endl;
  for (i=0; i<M.n(); i++) {
    for (j=0; j<M.m(); j++)
      f << std::setw(12) << M(i,j);
    f << "\n";
  }
  f << "\n\n";

  return f;
}
