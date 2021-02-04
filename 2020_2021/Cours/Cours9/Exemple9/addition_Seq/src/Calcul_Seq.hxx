#ifndef _CALCUL_HXX
#define _CALCUL_HXX

class Calcul_Seq {
public:
  Calcul_Seq(int n);
  ~Calcul_Seq();

  void init();
  void addition();
  double somme();

private:
  double * h_u, * h_v, * h_w;
  int n;
};

#ifndef IN_MAIN
extern 
#endif
int T_AllocId, T_CopyId, T_InitId, T_AddId, T_SommeId, T_FreeId;

#endif

