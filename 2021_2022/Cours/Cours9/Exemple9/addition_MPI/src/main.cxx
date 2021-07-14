#define IN_MAIN

#include <cstdlib>
#include <iostream>
#include "Calcul_MPI.hxx"
#include "timer.hxx"
#include <mpi.h>

int main(int argc, char **argv)
{  
  T_MPIId = AddTimer("mpi");
  T_AllocId = AddTimer("alloc");
  T_InitId = AddTimer("init");
  T_AddId = AddTimer("add");
  T_SommeId = AddTimer("somme");
  T_FreeId = AddTimer("free");
  AddTimer("total");

  int i, n = argc > 1 ? strtol(argv[1], NULL, 10) : 20000000;
  
  Timer & T_total = GetTimer(-1);
  T_total.start();
  
  Timer & T_mpi = GetTimer(-T_MPIId);
  T_mpi.start();

  MPI_Init(&argc, &argv);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  T_mpi.stop();

  int dn = (n + size)/size;
  int n0 = rank * dn, n1 = n0 + dn;
  if (rank == size-1) n1 = n;

  double v, v_local;
  Calcul_MPI C(n0, n1);
  C.init();
  C.addition();
  v_local = C.somme();

  T_mpi.start();
  MPI_Reduce(&v_local, &v, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  T_mpi.stop();
  
  T_total.stop();

  if (rank==0) {
    std::cout << "erreur " << v/n - 1.0 << "\n"
	            << std::endl;
  
    PrintTimers(std::cout);
  }
  
  return 0;
}
