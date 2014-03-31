/*
 *  Copyright (C) 2010-2011     CSC - IT Center for Science Ltd.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#ifdef PAPI
#include <papi.h>
#endif

#define gpts 20
#define nbands  600

/* Simple test for writing wafefunction-like data with MPI I/O
   Usage: mpirun -np 8 ./a.out 2 2 2 2
   The first parameter is the state parallelization, the following
   denote domain decomposition. Parallelization options have to be 
   compatible with the total number of processors */

int main(int argc, char *argv[]) {

  int my_id, nprocs;
  int mpi_dims[4]; 
  int period[4] = {0, 0, 0, 0};
  int coords[4];

  int dimsf[4] = {nbands, gpts, gpts, gpts};
  int count[4];
  int offset[4];
  int ndims = 4;

  double t0, t1;
#ifdef PAPI
  PAPI_dmem_info_t dmem;
  double mem1, mem2, mem1_max, mem2_max, mem1_ave, mem2_ave;
  int papi_err;
#endif

  double *my_data;

  MPI_Comm cart_comm;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  assert(argc == 5);
  for (int i=1; i < argc; i++)
    mpi_dims[i-1] = atoi(argv[i]);

  assert(mpi_dims[0] * mpi_dims[1] * mpi_dims[2] * mpi_dims[3] == nprocs);
  MPI_Cart_create(MPI_COMM_WORLD, 4, mpi_dims, period, 0, &cart_comm);
  MPI_Comm_rank(cart_comm, &my_id);

  MPI_Cart_coords(cart_comm, my_id, 4, coords);

  assert(nbands % mpi_dims[0] == 0);
  for (int i=1; i < 4; i++)
    assert(gpts % mpi_dims[i] == 0);

  int total_size = nbands*gpts*gpts*gpts;
  count[0] = nbands / mpi_dims[0];
  offset[0] = coords[0] * count[0];
  int data_size = count[0];
  for (int i=1; i < 4; i++)
    {
      count[i] = gpts/mpi_dims[i];
      offset[i] = coords[i] * count[i];
      data_size *= count[i];
    }

  my_data = (double *) malloc(data_size * sizeof(double));
  for (int i=0; i < data_size; i++)
    my_data[i] = my_id;

  MPI_Info info;
  MPI_File fp;
  MPI_Datatype filetype;
  // MPI_Info_set(info, "cb_nodes", "64");

  MPI_Barrier(MPI_COMM_WORLD);
#ifdef PAPI
  papi_err = PAPI_get_dmem_info(&dmem);
  if (papi_err != PAPI_OK)
    printf("PAPI_ERR\n");
  mem1 = (double)dmem.size / 1024.0;
  MPI_Reduce(&mem1, &mem1_max, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
  MPI_Reduce(&mem1, &mem1_ave, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
  mem1_ave /= nprocs;
#endif
  t0 = MPI_Wtime();
  MPI_File_open(MPI_COMM_WORLD, "test.dat",
                  MPI_MODE_CREATE|MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fp);

  MPI_Type_create_subarray(ndims, dimsf, count, offset, MPI_ORDER_C, 
			   MPI_DOUBLE, &filetype);
  MPI_Type_commit(&filetype);
  MPI_File_set_view(fp, 0, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);

  MPI_File_write_all(fp, my_data, data_size, MPI_DOUBLE, MPI_STATUS_IGNORE);

  MPI_Type_free(&filetype);
  MPI_File_close(&fp);
  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();
#ifdef PAPI
  papi_err = PAPI_get_dmem_info(&dmem);
  if (papi_err != PAPI_OK)
    printf("PAPI_ERR\n");
  mem2 = (double)dmem.size/ 1024.0;
  MPI_Reduce(&mem2, &mem2_max, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
  MPI_Reduce(&mem2, &mem2_ave, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
  mem2_ave /= nprocs;
#endif
  if (my_id == 0)
    {
      printf("IO time %f (%f) MB %f s\n", 
             total_size * 8/(1024.0*1024.0), 
             data_size * 8/(1024.0*1024.0), t1-t0);
#ifdef PAPI
      printf("Memory usage max (ave): %f (%f) %f (%f) \n", 
              mem1_max, mem1_ave, mem2_max, mem2_ave);
#endif
    }
      
  MPI_Finalize();
}
