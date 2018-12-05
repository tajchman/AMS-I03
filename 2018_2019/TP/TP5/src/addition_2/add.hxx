#include <vector>

void init_GPU    (double **u,
		  double **v,
		  size_t n);

void addition_GPU(std::vector<double> &w,
		  double * u,
		  double * v,
		  size_t n);
