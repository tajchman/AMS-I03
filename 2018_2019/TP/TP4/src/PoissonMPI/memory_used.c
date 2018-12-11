#include "memory_used.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

void memory_used()
{
  int rank, size;
  
  FILE* procfile;

  long to_read = 8192;
  char buffer[to_read];
  int read;

  long vmrss_kb, vmsize_kb;
  short found_vmrss = 0;
  short found_vmsize = 0;
  char* search_result;

  char delims[] = "\n";

  procfile = fopen("/proc/self/status", "r");
  read = fread(buffer, sizeof(char), to_read, procfile);
  fclose(procfile);
  char* line = strtok(buffer, delims);

  while (line != NULL && (found_vmrss == 0 || found_vmsize == 0) )
    {
      search_result = strstr(line, "VmRSS:");
      if (search_result != NULL)
        {
	  sscanf(line, "%*s %ld", &vmrss_kb);
	  found_vmrss = 1;
        }

      search_result = strstr(line, "VmSize:");
      if (search_result != NULL)
        {
	  sscanf(line, "%*s %ld", &vmsize_kb);
	  found_vmsize = 1;
        }

      line = strtok(NULL, delims);
    }

  
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (found_vmrss == 1 && found_vmsize == 1) {
    
    long vmrss_per_process[size];
    long vmsize_per_process[size];
  
    MPI_Gather(&vmrss_kb, 1, MPI_UNSIGNED_LONG, 
	       vmrss_per_process, 1, MPI_UNSIGNED_LONG, 
	       0, MPI_COMM_WORLD);
  
    MPI_Gather(&vmsize_kb, 1, MPI_UNSIGNED_LONG, 
	       vmsize_per_process, 1, MPI_UNSIGNED_LONG, 
	       0, MPI_COMM_WORLD);

    if (rank == 0) {
       long global_vmrss = 0;
       long global_vmsize = 0;
       printf("\n");
       for (int i = 0; i < size; i++)
	 {
	   printf("Process %03d: VmRSS = %6ld KB, VmSize = %6ld KB\n", 
                i, vmrss_per_process[i], vmsize_per_process[i]);
	   global_vmrss += vmrss_per_process[i];
	   global_vmsize += vmsize_per_process[i];
	 }
       printf("\nGlobal memory usage: VmRSS = %6ld KB, VmSize = %6ld KB\n\n", 
	      global_vmrss, global_vmsize);
     }

  }
}

