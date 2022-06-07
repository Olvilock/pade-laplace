#include <laplace/transform.cuh>

namespace numer {
	namespace laplace
	{
		using complex = thrust::complex<double>;
		
		namespace
		{
			__device__ void set_pow(double base)
			{
				extern __shared__ double pow[];
				pow[threadIdx.x] = 1.0;

				for (int exp = 1; exp < blockDim.x; exp <<= 1)
				{
					if (exp & threadIdx.x)
						pow[threadIdx.x] *= base;
					if (exp & (threadIdx.x + blockDim.x))
						pow[threadIdx.x + blockDim.x] *= base;
					base *= base;
				}
				__syncthreads();
			}

			template <int offset>
			__device__ complex get_sum(complex s, int power)
			{
				extern __shared__ double point_pow[];

				complex coeff = 1 / s;
				int next = 1;
#pragma unroll
				while (next <= offset)
					coeff *= next++ / s;

				auto res1 = coeff * point_pow[power];
				for (; power; ++next)
				{
					coeff *= -next / ((next - offset) * s);
					res1 /= power--;
					res1 += coeff * point_pow[power];
				}
				return res1;
			}
		}

		template<>
		__device__ thrust::pair<complex, complex> transform <transformType::Spline>(
			const thrust::complex<double> s,
			const SplineSegment* segments, unsigned segments_count,
			const SplineEndpoint left,
			const SplineEndpoint right)
		{
			if (!segments_count)
				return {};

			auto segment = *segments++;
			set_pow(-segment.pivot);

			auto coeff = thrust::exp(-segment.pivot * s);
			auto res1 = coeff *
				(get_sum<0>(s, threadIdx.x) * left.a +
					get_sum<1>(s, threadIdx.x) * left.b),
				res2 = coeff *
				(get_sum<1>(s, threadIdx.x + blockDim.x) * left.a +
					get_sum<2>(s, threadIdx.x + blockDim.x) * left.b);

			while (--segments_count)
			{
				auto old_cubic = segment.d;
				segment = *segments++;

				coeff *= (segment.d - old_cubic);
				res1 += coeff * get_sum<3>(s, threadIdx.x);
				res2 += coeff * get_sum<3>(s, threadIdx.x + blockDim.x);

				__syncthreads();
				set_pow(-segment.pivot);
				coeff = thrust::exp(-segment.pivot * s);
			}

			res1 -= coeff *
				(get_sum<0>(s, threadIdx.x) * right.a +
					get_sum<1>(s, threadIdx.x) * right.b +
					get_sum<3>(s, threadIdx.x) * segment.d);

			return { res1, res2 };
		}
	}
}