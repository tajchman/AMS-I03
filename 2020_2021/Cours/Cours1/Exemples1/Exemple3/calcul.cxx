#include "calcul.hxx"

void calcul1(std::vector<double> &u, std::vector<double> &v, std::vector<double> &w, 
             std::vector<double> &a, std::vector<double> &b, std::vector<double> &c, std::vector<double> &d)  {
    int i, n = u.size();

    for (i=0; i<n; i++) {
      u[i] = 2*a[i] + 3*b[i];
    }
    for (i=0; i<n; i++) {
      v[i] = 3*a[i] + 2*b[i];
      w[i] = c[i] + d[i];
    }
}

void calcul2(std::vector<double> &u, std::vector<double> &v, std::vector<double> &w, 
             std::vector<double> &a, std::vector<double> &b, std::vector<double> &c, std::vector<double> &d)  {
    int i, n = u.size();

    for (i=0; i<n; i++) {
      u[i] = 2*a[i] + 3*b[i];
      v[i] = 3*a[i] + 2*b[i];
    }
    for (i=0; i<n; i++) {
      w[i] = c[i] + d[i];
    }

}

