#include <vector>
#include <iostream>
#include <omp.h>
#include <cmath>
#include "somme.hxx"

static int cutoff;
void set_cutoff(int c)
{
   cutoff = c;
}

static double somme_seq_interne(const  std::vector<double> &v, int n1, int n2)
{
    int i;
    double r = 0;
    for (i=n1; i<n2; i++)
      r += exp(v[i]);

    return r;
}

static double somme_par_recursif(const  std::vector<double> &v, int n1, int n2)
{
#pragma omp critical
    std::cerr << "thread " << omp_get_thread_num() 
              << ": " << n1 << " - " << n2 << std::endl;

    // calcul sequentiel pour size(v) < 100
    if (n2 - n1 <= cutoff) {
        return somme_seq_interne(v, n1, n2);
    }

    // calcul sequentiel pour size(v) > 100
    double x, y;
    int n12 = (n1+n2)/2;

    #pragma omp task shared(x)
    x = somme_par_recursif(v, n1, n12);
    #pragma omp task shared(y)
    y = somme_par_recursif(v, n12, n2);
    #pragma omp taskwait
    x += y;

    return x;
}

double somme_par(const  std::vector<double> &v)
{
    double r;

    #pragma omp parallel
    #pragma omp single nowait
    r = somme_par_recursif(v, 0, v.size());
    return r;
}

double somme_seq(const  std::vector<double> &v)
{
    return somme_seq_interne(v, 0, v.size());
}

