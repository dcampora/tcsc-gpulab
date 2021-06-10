/**
   Most simple CUDA Hello World program

   author: Dorothea vom Bruch (dorothea.vom.bruch@cern.ch)
   date: 05/2019

 */

#include <stdio.h>
#include <iostream>

#include "../helpers/helpers.h"

using namespace std;

__global__ void hello_world_kernel() {

  /* blockIdx.x:  Accesses index of block within grid in x direction
     threadIdx.x: Accesses index of thread within block in x direction
   */
   if ( blockIdx.x < 100 && threadIdx.x < 100 ) 
    printf("Hello World from block %u, thread %u \n", blockIdx.x, threadIdx.x);
  
}

int main( int argc, char *argv[] ) {

  if ( argc != 3 ) {
    cout << "Need two arguments: number of blocks and number of threads" << endl;
    return -1;
  }

  const int n_blocks  = atoi(argv[argc-2]);
  const int n_threads = atoi(argv[argc-1]);
  
   
  /* dim3: CUDA specific variable to declare size of grid in blocks and threads, 
     can take up to three arguments for 3-dimensional grids and blocks
  */
  dim3 grid_dim(n_blocks);
  dim3 block_dim(n_threads);

  /* Syntax to launch a kernel: 
     <<< size of grid in blocks and threads>>>
     (): any parameters to be passed to the kernel
  */
  hello_world_kernel<<<grid_dim, block_dim>>>();

  /* Blocks until all requested tasks on device were completed;
     needed for printf in kernel to work
  */
  cudaDeviceSynchronize();

  return 0;
}
