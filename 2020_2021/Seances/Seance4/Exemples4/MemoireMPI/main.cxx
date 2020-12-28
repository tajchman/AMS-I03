#include <mpi.h>
#include "memory_used.hxx"

int main(int argc, char **argv)
{
  MemoryUsed M;

  M.initMeasure();
  MPI_Init(&argc, &argv);
  
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  M.setRank(rank, size);
  M.endMeasure("MPI_Init");
 
  M.initMeasure();

  MPI_Finalize();

  M.endMeasure("MPI_Finalize");

  if (rank == 0) {
    std::cout << size << " MPI processes:\n";
    int mean, stddev;
    GetMeasure("MPI_Init", size, mean, stddev);
    std::cout << "  MPI_Init :     " << mean << " +/- " << stddev << std::endl;
    GetMeasure("MPI_Finalize", size, mean, stddev);
    std::cout << "  MPI_Finalize : " << mean << " +/- " << stddev << std::endl;
  }
  return 0;
}
