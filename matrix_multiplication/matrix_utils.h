#pragma once

#include <string>
#include <vector>
#include <iostream>
#include "cuda_fp16.h"

template<typename T, typename U>
bool inline equals(T a, U b, double threshold) {
  auto diff = std::abs(double(a) - double(b));
  return diff < threshold;
}

/**
 * @brief Prints a matrix of size m x n.
 */
template<typename T>
void inline print_matrix(const T *A, const int m, const int n,
                         const std::string preamble = "",
                         const int max_columns = 0, const int max_rows = 0) {
  if (preamble != "") {
    std::cout << preamble << "\n";
  }
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if ((max_columns == 0 || j < max_columns) &&
          (max_rows == 0 || i < max_rows)) {
        printf("%10.4f   ", float(A[i * m + j]));
      }
    }
    if (max_rows == 0 || i < max_rows) {
      printf("\n");
    }
  }
  printf("\n\n");
}

/**
 * @brief Checks matrix A multiplied by matrix B results in C.
 *        Only checks elements until check_rows and check_columns
 */
template<typename TInput, typename TOutput>
void inline check_result(const TInput *A, const TInput *B, const TOutput *C,
                         const int m, const int n, const int k, const double threshold = 0.01,
                         const int check_rows = 64,
                         const int check_cols = 64) {
  TInput *test_matrix = new TInput[m * k];
  

  // Print matrices A, B, C
  // print_matrix(A, m, n, "Matrix A:");
  // print_matrix(B, n, k, "Matrix B:");
  // print_matrix(C, m, k, "Matrix C:");

  // Perform computation
  for (int row0 = 0; row0 < m && row0 < check_rows; ++row0) {
    for (int col1 = 0; col1 < k && col1 < check_cols; ++col1) {
      auto element = TInput(0);
      for (int i = 0; i < n; ++i) {
        element += A[row0 * m + i] * B[i * n + col1];
      }
      test_matrix[row0 * m + col1] = element;
    }
  }
  
  // print_matrix(test_matrix, m, k, "Test matrix:");

  // Check matrix matches
  // Print sub-matrix max 8x8
  int number_of_errors = 0;
  for (int i = 0; i < m && i < check_rows; ++i) {
    for (int j = 0; j < k && j < check_cols; ++j) {
      const auto cond = equals(C[i * m + j], test_matrix[i * m + j], threshold);
      number_of_errors += !cond;

      if (number_of_errors < 10 && !cond) {
        printf("Error in row %d col %d - Expected: %0.6f, Found: %0.6f\n", i,
               j, test_matrix[i * m + j], TInput(C[i * m + j]));
      }
    }
  }

  print_matrix(test_matrix, m, k, "Expected result (submatrix 8x8 max):", 8, 8);

  if (number_of_errors == 0) {
    printf("VERIFICATION PASSED\n");
  } else {
    printf("VERIFICATION FAILED\n");
  }

  delete[] test_matrix;
}
