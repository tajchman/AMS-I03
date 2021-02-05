#ifndef _CALCUL_MPI_Cuda_HXX
#define _CALCUL_MPI_Cuda_HXX

class Calcul_MPI_Cuda {
public:
  Calcul_MPI_Cuda(int m0, int m1, int rank);
  ~Calcul_MPI_Cuda();

  void init();
  void addition();
  double somme();

private:
  double * d_u, * d_v, * d_w, * d_tmp;
  int n0, n1;
  int n_local;
  int blockSize, gridSize;
};

#ifndef IN_MAIN
extern 
#endif
int T_CudaId, T_MPIId, T_AllocId, T_CopyId, T_InitId, T_AddId, T_SommeId, T_FreeId;

#endif

