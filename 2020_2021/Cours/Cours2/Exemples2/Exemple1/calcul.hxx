#include <vector>

void calcul_seq(std::vector<double> & v, 
                 double a, double (*f)(double, double),
                 const std::vector<double> & u);

void calcul_par0(std::vector<double> & v, 
                 double a, double (*f)(double, double),
                 const std::vector<double> & u);

void calcul_par1(std::vector<double> & v, 
                 double a, double (*f)(double, double),
                 const std::vector<double> & u);

void calcul_par2(std::vector<double> & v, 
                 double a, double (*f)(double, double),
                 const std::vector<double> & u);

void calcul_par3(std::vector<double> & v, 
                 double a, double (*f)(double, double),
                 const std::vector<double> & u);
