#ifndef _CALCUL_MPI_HXX
#define _CALCUL_MPI_HXX

class Calcul_MPI {
public:
  Calcul_MPI(int m0, int m1);
  ~Calcul_MPI();

  void init();
  void addition();
  double somme();

private:
  double * h_u, * h_v, * h_w;
  int n0, n1;
  int n_local;
};

#ifndef IN_MAIN
extern 
#endif
int T_MPIId, T_AllocId, T_CopyId, T_InitId, T_AddId, T_SommeId, T_FreeId;

#endif

