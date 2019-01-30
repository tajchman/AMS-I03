double * init(int n);
double * alloue(int n);
void libere(double **v);

void   iteration(double * v, const double * u, double dt, int n);
double difference(const double * u, const double * v, int n);
void save(const char *name, const double *u, int n);
