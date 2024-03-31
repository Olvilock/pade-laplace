#pragma once

#include <cuda/std/complex>

namespace pl {
__device__ cuda::std::complex<double> residue(const int dim);
} // namespace pl
