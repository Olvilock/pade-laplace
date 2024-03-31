#include <pl/fit.cuh>

namespace pl {
using complex = cuda::std::complex<double>;
namespace {
__device__ void set_pow(double base) {
  extern __shared__ double pow[];
  pow[threadIdx.x] = 1.0;
  for (int exp = 1; exp < blockDim.x; exp <<= 1) {
    if (exp & threadIdx.x)
      pow[threadIdx.x] *= base;
    if (exp & (threadIdx.x + blockDim.x))
      pow[threadIdx.x + blockDim.x] *= base;
    base *= base;
  }
  __syncthreads();
}

template <int offset> __device__ complex get_sum(complex s, int power) {
  extern __shared__ double point_pow[];
  complex coeff = 1.0 / s;
  int next = 1;
#pragma unroll
  while (next <= offset)
    coeff *= (double)next++ / s;

  auto res1 = coeff * point_pow[power];
  for (; power; ++next) {
    coeff *= (double)(-next) / ((double)(next - offset) * s);
    res1 /= power--;
    res1 += coeff * point_pow[power];
  }
  return res1;
}
} // namespace

__device__ cuda::std::pair<complex, complex>
transform(const complex s, cuda::std::span<const SplineSegment> segments,
          SplineEndpoint left, SplineEndpoint right) {
  if (segments.size() == 0)
    return {};
  auto segment = segments.begin();
  set_pow(-segment->pivot);

  auto coeff = cuda::std::exp(-segment->pivot * s);
  auto res1 = coeff * (get_sum<0>(s, threadIdx.x) * left.a +
                       get_sum<1>(s, threadIdx.x) * left.b),
       res2 = coeff * (get_sum<0>(s, threadIdx.x + blockDim.x) * left.a +
                       get_sum<1>(s, threadIdx.x + blockDim.x) * left.b);

  while (++segment != segments.end()) {
    auto old_cubic = segment->d;

    coeff *= (segment->d - old_cubic);
    res1 += coeff * get_sum<3>(s, threadIdx.x);
    res2 += coeff * get_sum<3>(s, threadIdx.x + blockDim.x);

    __syncthreads();
    set_pow(-segment->pivot);
    coeff = cuda::std::exp(-segment->pivot * s);
  }

  res1 -= coeff * (get_sum<0>(s, threadIdx.x) * right.a +
                   get_sum<1>(s, threadIdx.x) * right.b +
                   get_sum<3>(s, threadIdx.x) * segment->d);
  res2 -= coeff * (get_sum<0>(s, threadIdx.x + blockDim.x) * right.a +
                   get_sum<1>(s, threadIdx.x + blockDim.x) * right.b +
                   get_sum<3>(s, threadIdx.x + blockDim.x) * segment->d);
  return {res1, res2};
}
} // namespace pl
