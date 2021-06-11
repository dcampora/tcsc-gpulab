/**
   Most simple CUDA Hello World program

   author: Dorothea vom Bruch (dorothea.vom.bruch@cern.ch)
   date: 05/2019
   updated 06/2021
 */

#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void hello_world_gpu() {

  /* blockIdx.x:  Accesses index of block within grid in x direction
     threadIdx.x: Accesses index of thread within block in x direction
   */
   if ( blockIdx.x < 100 && threadIdx.x < 100 ) 
    printf("Hello World from the GPU at block %u, thread %u \n", blockIdx.x, threadIdx.x);
  
}

void hello_world_cpu() {
    printf("Hello World from the CPU \n");
}

int main( int argc, char *argv[] ) {

  if ( argc != 3 ) {
    cout << "Need two arguments: number of blocks and number of threads" << endl;
    return -1;
  }

  /* Call CPU function */
  hello_world_cpu();
    
  /* Call GPU function */
  const int n_blocks  = atoi(argv[argc-2]);
  const int n_threads = atoi(argv[argc-1]);
  
  dim3 grid_dim(n_blocks);
  dim3 block_dim(n_threads);

  hello_world_gpu<<<grid_dim, block_dim>>>();

  cudaDeviceSynchronize();

  return 0;
}
