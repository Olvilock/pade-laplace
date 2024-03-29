#pragma once

#include <cuda/std/complex>

namespace pl {
struct BatchTerm {
  cuda::std::complex<double> amp;
  cuda::std::complex<double> exp;
};

enum class BatchStatus {
  ok = 0b000,
  degenerate_system = 0b001,
  Aberth_divergence = 0b010,
  untouched = 0b100
};
__device__ const BatchStatus& operator |= (BatchStatus& me, const BatchStatus& other);

struct BatchResult {
  BatchTerm data;
  BatchStatus status;
};

template <typename Arg>
__global__ void kernelFitBatch(
  const cuda::std::complex<double>* grid,
  batchResult* result_grid,
  Arg arg);
} // namespace pl
