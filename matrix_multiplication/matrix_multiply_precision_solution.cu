/**
 Square matrix multiplication example

 author: Daniel Campora (dcampora@nvidia.com)
 date: 03/2024

*/

#include <chrono>
#include <cstdio>
#include <iostream>
#include "matrix_utils.h"

using storage_T = half;
using arithmetic_T = float;

// Define the tile size
constexpr int TILE_SIZE = 32;

/**
 * @brief Multiplies matrices using shared memory.
 * @details This last version of the square matrix multiplication uses
 *          shared memory and a predefined TILE_SIZE to preload data and
 *          speed up memory accesses.
 *
 *          Shared memory is populated in a coalesced manner, which more
 *          efficiently utilizes memory throughput.
 */
__global__ void multiply_square_matrices(const int size, const storage_T *A,
                                         const storage_T *B, storage_T *C) {
  // Sub-matrix this block works on
  int blockI = blockIdx.x;
  int blockJ = blockIdx.y;

  // Element within sub-matrix this thread works on
  int i = threadIdx.x;
  int j = threadIdx.y;

  // Define shared memory buffers which will be used to cache source matrices
  __shared__ storage_T shared_A[TILE_SIZE][TILE_SIZE];
  __shared__ storage_T shared_B[TILE_SIZE][TILE_SIZE];

  // Every thread calculates one element of destination sub-matrix
  arithmetic_T sum = 0;

  // Loop over blocks of size (TILE_SIZE x TILE_SIZE)
  for (int k = 0; k < (size / TILE_SIZE); ++k) {

    // Pointer to sub-matrix start
    const auto *sub_A = A + size * TILE_SIZE * blockI + TILE_SIZE * k;
    const auto *sub_B = B + size * TILE_SIZE * k + TILE_SIZE * blockJ;

    // Load sub-matrices into shared memory
    shared_A[i][j] = sub_A[i * size + j];
    shared_B[i][j] = sub_B[i * size + j];

    // Synchronize to make sure all threads have finished writing to shared
    // memory
    __syncthreads();

    // Multiply the two sub matrices
    for (int e = 0; e < TILE_SIZE; e++) {
      sum += static_cast<arithmetic_T>(shared_A[i][e]) * static_cast<arithmetic_T>(shared_B[e][j]);
    }

    // Synchronize to make sure all threads have computed the sum
    // before the next sub-matrices are loaded
    __syncthreads();
  }

  // Pointer to result sub-matrix
  auto *sub_C = C + size * TILE_SIZE * blockI + TILE_SIZE * blockJ;

  // Write result to global memory
  sub_C[i * size + j] = sum;
}

int main(int argc, char *argv[]) {

  if (argc != 2) {
    std::cout << "Needs an argument: number of rows (= number of columns) of "
                 "square matrices\n";
    return -1;
  }

  const int matrix_size = atoi(argv[argc - 1]);

  // Allocate host and device memory for three matrices
  storage_T *host_matrix[3]; // matrix[0] and matrix[1] are the source for the
                             // multiplication, result stored in matrix[2]
  storage_T *device_matrix[3];

  for (int i = 0; i < 3; i++) {
    host_matrix[i] = new storage_T[matrix_size * matrix_size];
    cudaMalloc((void **)&device_matrix[i],
               matrix_size * matrix_size * sizeof(storage_T));
  }

  // Initialize matrices
  for (int i = 0; i < matrix_size; i++) {
    for (int j = 0; j < matrix_size; j++) {
      host_matrix[0][i * matrix_size + j] = 0.1 * (((i + 1) * (j + 1)) % 10);
      host_matrix[1][i * matrix_size + j] = 0.1 * ((2 * i + j) % 10);
      host_matrix[2][i * matrix_size + j] = 0;
    }
  }
     
  // Copy matrices to device
  for (int i = 0; i < 3; i++) {
    cudaMemcpy(device_matrix[i], host_matrix[i],
               matrix_size * matrix_size * sizeof(storage_T),
               cudaMemcpyHostToDevice);
  }

  // Launch kernel
  int size = matrix_size;
  int number_of_threads = TILE_SIZE;
  int number_of_blocks = (size + number_of_threads - 1) / number_of_threads;

  dim3 grid(number_of_blocks, number_of_blocks);
  dim3 block(number_of_threads, number_of_threads);

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  multiply_square_matrices<<<grid, block>>>(size, device_matrix[0],
                                            device_matrix[1], device_matrix[2]);

  cudaDeviceSynchronize();

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;

  // Copy back result
  cudaMemcpy(host_matrix[2], device_matrix[2],
             matrix_size * matrix_size * sizeof(storage_T), cudaMemcpyDeviceToHost);

  // Check and print result
  double threshold = 0.1;
  std::vector<double> host_matrix_A_d(matrix_size * matrix_size);
  std::vector<double> host_matrix_B_d(matrix_size * matrix_size);
  for (int i = 0; i < matrix_size; i++) {
    for (int j = 0; j < matrix_size; j++) {
      host_matrix_A_d[i * matrix_size + j] = 0.1 * (((i + 1) * (j + 1)) % 10);
      host_matrix_B_d[i * matrix_size + j] = 0.1 * ((2 * i + j) % 10);
    }
  }
  check_result<double>(host_matrix_A_d.data(), host_matrix_B_d.data(), host_matrix[2], matrix_size,
                       matrix_size, matrix_size, threshold);

  std::cout << "Kernel duration: " << elapsed_seconds.count() << " s\n";

  // Free memory
  for (int i = 0; i < 3; i++) {
    delete[] host_matrix[i];
    cudaFree(device_matrix[i]);
  }
}
