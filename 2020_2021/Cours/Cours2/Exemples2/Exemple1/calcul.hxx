#include <vector>

double calcul_seq(std::vector<double> & v, 
                 double a, double (*f)(double, double),
                 const std::vector<double> & u);

double calcul_par0(std::vector<double> & v, 
                 double a, double (*f)(double, double),
                 const std::vector<double> & u,
                 const std::vector<double> & v_seq);

double calcul_par1(std::vector<double> & v, 
                 double a, double (*f)(double, double),
                 const std::vector<double> & u,
                 const std::vector<double> & v_seq);

double calcul_par2(std::vector<double> & v, 
                 double a, double (*f)(double, double),
                 const std::vector<double> & u,
                 const std::vector<double> & v_seq);

double calcul_par3(std::vector<double> & v, 
                 double a, double (*f)(double, double),
                 const std::vector<double> & u,
                 const std::vector<double> & v_seq);
