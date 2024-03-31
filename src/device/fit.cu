#include "polyroot.cuh"
#include "residue.cuh"
#include "taylor_slae.cuh"

#include <pl/fit.cuh>

#include <cuda/std/complex>
#include <cuda/std/span>
#include <cuda/std/tuple>

namespace pl {
using complex = cuda::std::complex<double>;

namespace {
__device__ void
fitBatch(const complex point, BatchResult *out_ptr,
         const cuda::std::pair<complex, complex> transform_pair) {
  extern __shared__ complex roots[];
  auto coeff = roots + blockDim.x;
  auto taylor = coeff + blockDim.x;

  taylor[threadIdx.x] = transform_pair.first;
  taylor[threadIdx.x + blockDim.x] = transform_pair.second;

  // printf("%i: %le %le\n", threadIdx.x,
  //	taylor[threadIdx.x].real(), taylor[threadIdx.x].imag());

  for (int lc_dim = 1; lc_dim <= blockDim.x; out_ptr += lc_dim++) {
    BatchStatus status = BatchStatus::ok;
    __syncthreads();

    if (!(slaeCG(lc_dim, 1000) < 1e-6))
      status |= BatchStatus::degenerate_system;

    auto this_coeff = taylor[threadIdx.x];
    if (threadIdx.x < lc_dim && threadIdx.x) {
      int id_sum = threadIdx.x - 1;
      int id = id_sum / 2;

      if (id_sum % 2)
        this_coeff += taylor[id_sum - id] * coeff[id];
      this_coeff += taylor[id] * coeff[id_sum - id];

      while (id--)
        this_coeff +=
            taylor[id] * coeff[id_sum - id] + taylor[id_sum - id] * coeff[id];
    }

    auto highest_coeff = coeff[threadIdx.x];

    if (!(solveAberth(lc_dim, 100) < 1e-6))
      status |= BatchStatus::Aberth_divergence;

    if (threadIdx.x == lc_dim - 1)
      coeff[0] = highest_coeff;
    __syncthreads();
    for (int id = 1; id < lc_dim; id <<= 1) {
      if (threadIdx.x < id && threadIdx.x + id < lc_dim)
        coeff[threadIdx.x + id] = coeff[threadIdx.x];
      __syncthreads();
    }
    highest_coeff = coeff[threadIdx.x];

    coeff[threadIdx.x] = this_coeff;
    __syncthreads();

    if (threadIdx.x < lc_dim)
      out_ptr[threadIdx.x] = {
          {residue(lc_dim) / highest_coeff, roots[threadIdx.x] + point},
          status};
  }
}
} // namespace

template __global__ void fitTransform(const complex *grid,
                                      BatchResult *result_grid,
                                      cuda::std::span<const DeviceNode>);
template __global__ void
fitTransform(const complex *grid, BatchResult *result_grid,
             cuda::std::span<const SplineSegment> segments, SplineEndpoint left,
             SplineEndpoint right);

template <typename... TransformArgs>
__global__ void fitTransform(const complex *grid, BatchResult *result_grid,
                             TransformArgs... args) {
  const auto point = grid[blockIdx.x];
  fitBatch(point, result_grid + blockIdx.x * blockDim.x * (blockDim.x + 1) / 2,
           transform(point, args...));
}
} // namespace pl
