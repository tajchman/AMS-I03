#include <mpi.h>
#include <stdio.h>
#include <omp.h>

int main(int argc, char **argv)
{
  char name[1024];
  int res;

  int provided, required = MPI_THREAD_MULTIPLE;

  MPI_Init_thread( &argc, &argv, required, &provided );

  printf ("required = %d provided = %d\n", required, provided);

  #pragma omp parallel
  #pragma omp master
    printf("%d threads\n", omp_get_num_threads());


  MPI_Get_processor_name(name, &res );
  MPI_Finalize();

  name[res] = '\0';
  printf("proc name = %s\n", name);
  return 0;
}
