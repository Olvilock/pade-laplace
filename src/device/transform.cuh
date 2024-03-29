#pragma once

#include "solver.h"

#include <cuda/std/tuple>
#include <cuda/std/complex>
#include <cuda/std/span>

namespace pl {
template <typename Arg>
__device__ cuda::std::pair<cuda::std::complex<double>, cuda::std::complex<double>>
transform(const cuda::std::complex<double> s, Arg arg);

struct DeviceNode {
  double point;
  cuda::std::complex<double> value;

  DeviceNode() = default;
  __host__ __device__ DeviceNode(const pl::Node& node);
};

using TrapeziaData = std::span<DeviceNode>;
struct SplineData {
  struct SplineSegment {
    double pivot;
    cuda::std::complex<double> d;
  };
  struct SplineEndpoint {
    cuda::std::complex<double> a, b;
  };

  cuda::std::span<SplineSegment> segments;
  const SplineEndpoint left, right;
};
} // namespace pl
