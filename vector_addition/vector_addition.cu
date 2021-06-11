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

/* Step 2 to do: Add the __global__ keyword in front of the function declaration  */
void vector_addition_gpu(int *a, int *b, int *c, int size) {
  
  /* Step 2 to do: modify the for loop such that it is executed in parallel
     Follow the instructions in the notebook for details
     The idea is to use threadIdx.x, blockIdx.x, blockDim.x and gridDim.x 
     to configure the start and stride of the for loop
   */
  const int start = 0;
  const int stride = 1;
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
  
  /* Allocate device memory */
  /* Step 1 to do: Allocate global memory for the device variables
     Use the following syntax, which allocates size * sizeof(int) bytes and stores the pointer in *a_d:
     cudaMalloc( (void**)&a_d, size * sizeof(int) );
     Do the same for b_d and c_d!
     */
  
  
  /* Copy host vectors to the device */
  /* Step 1 to do: Copy the host vectors to the device
     Use the following syntax:
     cudaMemcpy( a_d, a_h, size * sizeof(int), cudaMemcpyHostToDevice );
     This copies size * sizeof(int) bytes from a_h to a_d, copying from the host to the device
     Do the same to copy b_h to b_d and c_h to c_d
     Note that we also copy vector c, such that it is initialized to zero. 
     There other methods available in CUDA to do that, but for simplicity we do it this way.
   */

  
  
  /* Define grid dimensions */
  /* Step 2 to do: define the dim3 variables for the grid dimensions as follows:
     dim3 grid_dim(n_blocks);
     dim3 block_dim(n_threads);
  */


  /* Call kernel */
  /* Step 2 to do: Launch the kernel 
     Call the kernel vector_addition_gpu with the grid dimension settings and passing the device vector pointers and the size as input
   */
  
  
  /* Copy result vector from device to host */
  /* Step 3 to do: Copy content of c_d to c_h 
     Follow the syntax of cudaMemcpy from above, but remember that we want to copy from teh device to the host now
   */

  
  /* Make sure GPU work is done */
  /* Step 3 to do: Call the same synchronization function we used in the hello_world example to ensure that the GPU work has finished */

  /* Step 3 to do: Uncomment the call to check_elements to verify the vector addition result from the GPU  */
  //check_elements(22, c_h, size);
 
  /* Step 1 to do: Free device memory
     Use the following syntax:
     cudaFree( a_d );
     This will free the global memory previously allocated with cudaMalloc
     Do the same for b_d and c_d!
   */
  

  /* free host memory */
  delete [] a_h;
  delete [] b_h;
  delete [] c_h;

  
  return 0;
}
