/**
   Vector addition:
   takes vectors a and b as input, computes vector sum 
   and stores output in vector c

   author: Dorothea vom Bruch (dorothea.vom.bruch@cern.ch)
   date: 05/2019
   updated: 06/2021

 */

#include <stdio.h>
#include <iostream>

using namespace std;

void init_with(int* a, float val, int N) {
    for ( int i = 0; i < N; i++ ) {
        a[i] = val;
    }
}

void vector_addition_cpu(int *a, int *b, int *c, int N) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

void check_elements(int correct, int* vec, int N) {
    for (int i = 0; i < N; i++) {
         if (vec[i] != correct) {
                 printf("ERROR: vec[%u] = %d, should be %d \n", i, vec[i], correct);
         }
    }
}

__global__ void vector_addition_gpu(int *a, int *b, int *c, int size) {
  const int start = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = start; i < size; i += stride) {
    c[i] = a[i] + b[i];
  }
}

int main(int argc, char *argv[] ) {

  if ( argc != 4 ) {
    cout << "Need three arguments: size of vector, number of threads / block and number of blocks in the grid" << endl;
    return -1;
  }
  
  const int size  = atoi(argv[argc-3]);
  const int n_threads = atoi(argv[argc-2]);
  const int n_blocks = atoi(argv[argc-1]);

  cout << "Adding vectors of size " <<  size << " with " << n_threads << " threads" << " and " << n_blocks << " blocks" << endl;  
 
  /* Host memory for the two input vectors a and b and the output vector c */
  int *a_h = new int[size];
  int *b_h = new int[size];
  int *c_h = new int[size];

  /* Initialize vectors */
  init_with(a_h, 13, size);
  init_with(b_h, 9, size);
  init_with(c_h, 0, size);
  
  /* Device pointers for the three vectors a, b, c */
  int *a_d, *b_d, *c_d;
  cudaMalloc( (void**)&a_d, size * sizeof(int) );
  cudaMalloc( (void**)&b_d, size * sizeof(int) );
  cudaMalloc( (void**)&c_d, size * sizeof(int) );

  /* Copy vectors to device */
  cudaMemcpy( a_d, a_h, size * sizeof(int), cudaMemcpyHostToDevice );
  cudaMemcpy( b_d, b_h, size * sizeof(int), cudaMemcpyHostToDevice );
  cudaMemcpy( c_d, c_h, size * sizeof(int), cudaMemcpyHostToDevice );
  
  /* Define grid dimensions */
  dim3 grid_dim(n_blocks);
  dim3 block_dim(n_threads);

  /* Call kernel */
  vector_addition_gpu<<<grid_dim, block_dim>>>( a_d, b_d, c_d, size);

  cudaMemcpy( c_h, c_d, size * sizeof(int), cudaMemcpyDeviceToHost );

  /* Make sure GPU work is done */
  cudaDeviceSynchronize();

  check_elements(22, c_h, size);

  cudaFree( a_d );
  cudaFree( b_d );
  cudaFree( c_d );

  /* free host memory */
  delete [] a_h;
  delete [] b_h;
  delete [] c_h;

  
  return 0;
}
