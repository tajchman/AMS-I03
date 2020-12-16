#include <mpi.h>
#include "memory_used.hxx"

int main(int argc, char **argv)
{
  MemoryUsed M;

  M.initMeasure();
  MPI_Init(&argc, &argv);
  
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  M.setRank(rank, size);
  M.endMeasure("MPI_Init");
 
  M.initMeasure();

  MPI_Finalize();
  
  M.endMeasure("MPI_Finalize");
  return 0;
}
