/**
   Most simple CUDA Hello World program

   author: Dorothea vom Bruch (dorothea.vom.bruch@cern.ch)
   date: 05/2019
   updated 06/2021
 */

#include <stdio.h>
#include <iostream>

using namespace std;

/* to do: Add the __global__ keyword in front of the function declaration to indicate that this function is executed on the GPU */
void hello_world_gpu() {
    
    /* to do: uncomment this line once the hello_world_gpu function has been marked with the __global__ keyword */
    //printf("Hello World from the GPU at block %u, thread %u \n", blockIdx.x, threadIdx.x);
  
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
  
   
  /* Refactor and code below to call the function on the GPU */  
    
  /* to do: variables of type dim3 to declare the size of the grid (n_blocks) and the size of the blocks (n_threads) 
      example: dim3 grid_dim(n_blocks);
  */

    
  /* to do: launch the kernel
     Reminder: Syntax to launch a kernel: 
     hello_world_gpu<<<grid_dim, block_dim>>>();
     grid_dim and block_dim are the variables of type dim3 that you declared above
     paramters can be passed to the function in the brackets (), we leave them empty for this exercise
  */

  /* to do: call the pre-defined function cudaDeviceSynchronize();
     It blocks until all requested tasks on device were completed
  */
  

  return 0;
}
