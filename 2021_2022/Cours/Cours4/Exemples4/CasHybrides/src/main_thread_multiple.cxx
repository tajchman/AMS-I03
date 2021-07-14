#include <iostream>
#include <vector>
#include <mpi.h>
#include <omp.h>

#define MAX_MESSAGE 1024

int main(int argc, char **argv)
{
  int provided;
  int requested = MPI_THREAD_MULTIPLE;

  MPI_Init_thread(&argc, &argv, requested, &provided);

  std::cerr << "requested : " << requested 
            << " provided : " << provided << std::endl;

  if(provided < requested) {
    std::cerr << "support MPI-OpenMP insuffisant\n";
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  else
  {
    std::cerr << "support MPI-OpenMP ok\n";
  }

  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::cout << "\nProcessus " << rank << "/" << size << std::endl << std::endl;

  char tampon[MAX_MESSAGE];

  if (rank == 0) {
    sprintf(tampon, "ici processus %d (région séquentielle)", rank);
    std::cerr << "in process " << rank 
              << ", message envoye '" << tampon << "'\n" << std::endl;
    MPI_Send(tampon, MAX_MESSAGE, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
  }
  if (rank == 1) {
    MPI_Status status;
    MPI_Recv(tampon, MAX_MESSAGE, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
    std::cerr << "in process " << rank 
              << ", message recu '" << tampon << "'\n" << std::endl;
  }

  omp_set_num_threads(2);

#pragma omp parallel private(tampon)
  {
    int iThread = omp_get_thread_num();
    if (rank == 0) {
      sprintf(tampon, "ici processus %d thread %d", 
              rank, iThread);
              
#pragma omp critical
      {
      std::cerr << "in process " << rank << " thread " << iThread 
                << ", message envoye '" << tampon << "'" << std::endl;
      }

      MPI_Send(tampon, MAX_MESSAGE, MPI_CHAR, 1, iThread, MPI_COMM_WORLD);
    }
    if (rank == 1) {
      MPI_Status status;
      MPI_Recv(tampon, MAX_MESSAGE, MPI_CHAR, 0, iThread, MPI_COMM_WORLD, 
               &status);

#pragma omp critical
      {
      std::cerr << "in process " << rank << " thread " << iThread 
                << ", message recu '" << tampon << "'" << std::endl;
      }
    }
  }

  MPI_Finalize();
  std::cerr << "Fin du traitement" << std::endl;
  return EXIT_SUCCESS;
}
