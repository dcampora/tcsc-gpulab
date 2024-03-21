/**
 Square matrix multiplication example

 author: Daniel Campora (dcampora@nvidia.com)
 date: 03/2024

*/

#include <chrono>
#include <cstdio>
#include <iostream>
#include "matrix_utils.h"
#include <mma.h>
using namespace nvcuda;

using storage_T = half;
using arithmetic_T = float;

// Define the tile size
constexpr int TILE_SIZE = 16;

/**
 * @brief Multiplies matrices using Tensor Cores.
 * @details Uses Tensor Cores to perform matrix-matrix multiplication.
 *          Each warp should work on a separate TILE_SIZE * TILE_SIZE fragment.
 */
__global__ void multiply_square_matrices(const int size, const storage_T *A,
                                         const storage_T *B, arithmetic_T *C) {
  // Sub-matrix this warp works on
  int blockI = blockIdx.x * blockDim.z + threadIdx.z;
  int blockJ = blockIdx.y * blockDim.y + threadIdx.y;

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, storage_T, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, storage_T, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, arithmetic_T> c_frag;
  
  // Initialize the output to zero
  wmma::fill_fragment(c_frag, 0);

  // Loop over blocks of size (TILE_SIZE x TILE_SIZE)
  for (int k = 0; k < (size / TILE_SIZE); ++k) {

    // Pointer to sub-matrix start
    const auto *sub_A = A + size * TILE_SIZE * blockI + TILE_SIZE * k;
    const auto *sub_B = B + size * TILE_SIZE * k + TILE_SIZE * blockJ;

    // Load the inputs
    wmma::load_matrix_sync(a_frag, sub_A, size);
    wmma::load_matrix_sync(b_frag, sub_B, size);

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  // Pointer to result sub-matrix
  auto *sub_C = C + size * TILE_SIZE * blockI + TILE_SIZE * blockJ;

  // Store the output
  wmma::store_matrix_sync(sub_C, c_frag, size, wmma::mem_row_major);
}

int main(int argc, char *argv[]) {
  
  if (argc != 2) {
    std::cout << "Needs an argument: number of rows (= number of columns) of "
                 "square matrices\n";
    return -1;
  }

  const int matrix_size = atoi(argv[argc - 1]);
  
  // Allocate host and device memory for three matrices
  storage_T *host_matrix[2];            // matrix[0] and matrix[1] are the source for the
  arithmetic_T *host_result_matrix; // multiplication, result stored in host_result_matrix
  storage_T *device_matrix[2];
  arithmetic_T *device_result_matrix;

  for (int i = 0; i < 2; i++) {
    host_matrix[i] = new storage_T[matrix_size * matrix_size];
    cudaMalloc((void **)&device_matrix[i],
               matrix_size * matrix_size * sizeof(storage_T));
  }
  host_result_matrix = new arithmetic_T[matrix_size * matrix_size];
  cudaMalloc((void **)&device_result_matrix,
             matrix_size * matrix_size * sizeof(arithmetic_T));

  // Initialize matrices
  for (int i = 0; i < matrix_size; i++) {
    for (int j = 0; j < matrix_size; j++) {
      host_matrix[0][i * matrix_size + j] = 0.1 * (((i + 1) * (j + 1)) % 10);
      host_matrix[1][i * matrix_size + j] = 0.1 * ((2 * i + j) % 10);
      host_result_matrix[i * matrix_size + j] = 0;
    }
  }
     
  // Copy input matrices to device
  for (int i = 0; i < 2; i++) {
    cudaMemcpy(device_matrix[i], host_matrix[i],
               matrix_size * matrix_size * sizeof(storage_T),
               cudaMemcpyHostToDevice);
  }
  cudaMemcpy(device_result_matrix, host_result_matrix,
             matrix_size * matrix_size * sizeof(arithmetic_T),
             cudaMemcpyHostToDevice);

  // Launch kernel
  int size = matrix_size;
  int number_of_threads_y = 4;
  int number_of_threads_z = 4;
  int number_of_blocks = size / (TILE_SIZE * number_of_threads_y * number_of_threads_z);

  dim3 grid(number_of_blocks, number_of_blocks);
  dim3 block(32, number_of_threads_y, number_of_threads_z);

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  multiply_square_matrices<<<grid, block>>>(size, device_matrix[0],
                                            device_matrix[1], device_result_matrix);

  cudaDeviceSynchronize();

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;

  // Copy back result
  cudaMemcpy(host_result_matrix, device_result_matrix,
             matrix_size * matrix_size * sizeof(arithmetic_T), cudaMemcpyDeviceToHost);

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
  check_result<double>(host_matrix_A_d.data(), host_matrix_B_d.data(), host_result_matrix, matrix_size,
                       matrix_size, matrix_size, threshold);

  std::cout << "Kernel duration: " << elapsed_seconds.count() << " s\n";

  // Free memory
  for (int i = 0; i < 2; i++) {
    delete[] host_matrix[i];
    cudaFree(device_matrix[i]);
  }
  delete[] host_result_matrix;
  cudaFree(device_result_matrix);
}
