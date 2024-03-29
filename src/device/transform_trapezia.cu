#include "transform.cuh"

namespace pl {
	using complex = cuda::std::complex<double>;
	namespace {
		__device__ cuda::std::pair<double, double> get_pow(double base) {
			double res1 = 1.0, res2 = 1.0;
			for (int exp = 1; exp < 2 * blockDim.x; exp <<= 1) {
				if (exp & threadIdx.x) res1 *= base;
				if (exp & (threadIdx.x + blockDim.x)) res2 *= base;
				base *= base;
			}
			return { res1, res2 };
		}
	}

	template<>
	__device__ cuda::std::pair<complex, complex> transform (
		const cuda::std::complex<double> s,
		TrapeziaData data)
	{
		complex	res1 = {}, res2 = {};
		if (data.count) {
			auto right = *data.nodes++;
			auto pow = get_pow(-right.point);
				
			auto left_c1 = right.value * thrust::exp(-right.point * s);
			auto left_c2 = pow.second * left_c1;
			left_c1 *= pow.first;
				
			while (--data.count) {
				auto left_point = right.point;
				right = *data.nodes++;
				pow = get_pow(-right.point);
					
				auto right_c1 = right.value * thrust::exp(-right.point * s);
				auto right_c2 = pow.second * right_c1;
				right_c1 *= pow.first;

				res1 += 0.5 * (right.point - left_point) * (left_c1 + right_c1);
				res2 += 0.5 * (right.point - left_point) * (left_c2 + right_c2);

				left_c1 = right_c1;
				left_c2 = right_c2;
			}
		}
		int i = 2;
		while (i <= threadIdx.x) {
			res1 /= i;
			res2 /= i++;
		}
		while (i <= threadIdx.x + blockDim.x) res2 /= i++;
		return { res1, res2 };
	}
}
