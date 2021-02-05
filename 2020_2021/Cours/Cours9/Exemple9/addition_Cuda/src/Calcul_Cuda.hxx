#ifndef _CALCUL_HXX
#define _CALCUL_HXX

class Calcul_Cuda {
public:
  Calcul_Cuda(int n);
  ~Calcul_Cuda();

  void init();
  void addition();
  double somme();

private:
  int n;
  double *d_u, *d_v, *d_w, *d_tmp;
  unsigned int blockSize, gridSize;
};

#ifndef IN_MAIN
extern 
#endif
int T_AllocId, T_CopyId, T_InitId, T_AddId, T_SommeId, T_FreeId;

#endif

