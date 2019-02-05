class Matrix;

void Iteration(Matrix &v, const Matrix &u, const Matrix &f,
	       double lambda, double dt);

double Difference(const Matrix &v, const Matrix &u);
