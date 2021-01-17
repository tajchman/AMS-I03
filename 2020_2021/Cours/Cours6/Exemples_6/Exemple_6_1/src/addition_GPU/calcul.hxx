#ifndef _CALCUL_HXX
#define _CALCUL_HXX

class Calcul_GPU {
public:
  Calcul_GPU(int n);
  ~Calcul_GPU();

  void init();
  void addition();
  double verification();

private:
  int n;
  double *d_u, *d_v, *d_w;
  unsigned int blockSize, gridSize;
};

#ifndef IN_MAIN
extern 
#endif
int T_AllocId, T_CopyId, T_InitId, T_AddId, T_VerifId, T_FreeId;

#endif

