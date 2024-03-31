#pragma once

#include <pl/fit.h>

#include <cuda/std/complex>
#include <cuda/std/span>
#include <cuda/std/tuple>

namespace pl {
struct BatchTerm {
  cuda::std::complex<double> coeff;
  cuda::std::complex<double> exp;
};

enum class BatchStatus {
  ok = 0b000,
  degenerate_system = 0b001,
  Aberth_divergence = 0b010,
  untouched = 0b100,
};
__device__ const inline BatchStatus &operator|=(BatchStatus &me,
                                                const BatchStatus &other) {
  me = static_cast<BatchStatus>(static_cast<int>(me) | static_cast<int>(other));
  return me;
}

struct BatchResult {
  BatchTerm data;
  BatchStatus status;
};

struct DeviceNode {
  double point;
  cuda::std::complex<double> value;

  DeviceNode() = default;
  __host__ __device__ inline DeviceNode(const pl::Node &node)
      : point{node.point}, value{node.value} {}
};

struct SplineSegment {
  double pivot;
  cuda::std::complex<double> d;
};
struct SplineEndpoint {
  cuda::std::complex<double> a, b;
};

template <typename... TransformArgs>
__global__ void fitTransform(const cuda::std::complex<double> *grid,
                             BatchResult *result_grid, TransformArgs... arg);

__device__
    cuda::std::pair<cuda::std::complex<double>, cuda::std::complex<double>>
    transform(const cuda::std::complex<double> s,
              cuda::std::span<const DeviceNode> data);

__device__
    cuda::std::pair<cuda::std::complex<double>, cuda::std::complex<double>>
    transform(const cuda::std::complex<double> s,
              cuda::std::span<const SplineSegment> segments,
              SplineEndpoint left, SplineEndpoint right);
} // namespace pl
