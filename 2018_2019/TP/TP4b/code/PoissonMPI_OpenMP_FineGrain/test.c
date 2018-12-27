#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv)
{
  int rank, size;
  int l;
  char s[MPI_MAX_PROCESSOR_NAME];

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, & rank);
  MPI_Comm_size(MPI_COMM_WORLD, & size);
  MPI_Get_processor_name(s, &l);
  s[l] = '\0';

  fprintf(stderr, "%d / %d %s\n", rank, size, s);
  MPI_Finalize();
  return 0;
}
