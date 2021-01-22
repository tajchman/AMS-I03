#ifndef _CALCUL_HXX
#define _CALCUL_HXX

class Calcul_CPU {
public:
  Calcul_CPU(int n);
  ~Calcul_CPU();

  void init();
  void addition();
  double verification();

private:
  double * h_u, * h_v, * h_w;
  int n;
};

#ifndef IN_MAIN
extern 
#endif
int T_AllocId, T_CopyId, T_InitId, T_AddId, T_VerifId, T_FreeId;

#endif

