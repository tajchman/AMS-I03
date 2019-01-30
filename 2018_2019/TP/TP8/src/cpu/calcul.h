double * alloue_work(int n);
double * alloue(int n);
void libere(double **v);

double * init     (int n);
double * zero     (int n);
 
void laplacien    (double * u_diffuse,
                   const double * u_current,
                   double dx, int n);

void calcul_forces(double * forces,
                   const double * u_current,
                   int n);

void variation    (double * u_next,
                   const double * u_current,
                   const double * u_diffuse,
                   const double * forces,
                   double dt, int n);


double difference (const double * u,
                   const double * v,
		   double * work,
                   int n);

void save(int k,
          const double *u,
          int n);
