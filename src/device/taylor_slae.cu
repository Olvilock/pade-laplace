#include "taylor_slae.cuh"
#include <cuda/std/complex>

namespace pl {
using complex = cuda::std::complex<double>;
namespace {
__device__ void apply(const int dim, complex vec) {
  extern __shared__ complex error[];
  complex *roots = error + blockDim.x;
  const complex *taylor = roots + blockDim.x;

  roots[threadIdx.x] = {};
  __syncthreads();
  if (threadIdx.x < dim)
    for (int id = threadIdx.x + 1;; ++id) {
      id -= dim * (id == dim);
      roots[id] += vec * taylor[id + threadIdx.x];

      if (id == threadIdx.x)
        break;
    }
  __syncthreads();
}

__device__ void reduce(const int dim) {
  extern __shared__ double data[];
#pragma unroll
  for (int id = 512; id; id >>= 1) {
    if (threadIdx.x + id < dim && threadIdx.x < id)
      data[threadIdx.x] += data[threadIdx.x + id];
    __syncthreads();
  }
#pragma unroll
  for (int id = 1; id < blockDim.x; id <<= 1) {
    if (threadIdx.x < id && threadIdx.x + id < blockDim.x)
      data[threadIdx.x + id] = data[threadIdx.x];
    __syncthreads();
  }
}
} // namespace

__device__ double slaeCG(const int dim, int iter_count) {
  extern __shared__ complex buffer[];
  complex *result = buffer + blockDim.x;
  const complex *minus_rhs = result + blockDim.x + dim;

  static_assert(sizeof(complex) >= sizeof(double),
                "sizeof(complex) must not be less than sizeof(double)");
  extern __shared__ double norm[];

  __syncthreads();
  apply(dim, cuda::std::conj(-minus_rhs[threadIdx.x]));
  norm[threadIdx.x] = cuda::std::norm(result[threadIdx.x]);
  __syncthreads();
  reduce(dim);

  complex error = cuda::std::conj(result[threadIdx.x]), basis = error, root{};
  auto err_norm = norm[threadIdx.x];
  while (iter_count--) {
    if (err_norm < 1e-64)
      break;

    __syncthreads();
    apply(dim, basis);
    if (threadIdx.x < dim)
      norm[threadIdx.x] = cuda::std::norm(result[threadIdx.x]);
    __syncthreads();
    reduce(dim);
    apply(dim, cuda::std::conj(result[threadIdx.x]));

    auto alpha = err_norm / norm[threadIdx.x];
    root += alpha * basis;
    error -= alpha * cuda::std::conj(result[threadIdx.x]);
    if (threadIdx.x < dim)
      norm[threadIdx.x] = cuda::std::norm(error);
    __syncthreads();
    reduce(dim);

    auto beta = norm[threadIdx.x] / err_norm;
    basis = error + beta * basis;
    err_norm = norm[threadIdx.x];
  }

  if (threadIdx.x < dim)
    result[dim - threadIdx.x - 1] = root;
  __syncthreads();

  return sqrt(err_norm);
}
} // namespace pl
