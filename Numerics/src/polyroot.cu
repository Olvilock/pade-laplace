#include <polyroot.cuh>

#include <thrust/complex.h>

namespace numer
{
	using complex = thrust::complex<double>;
	
	namespace
	{
		__device__ complex evalDelta(const int dim)
		{
			extern __shared__ complex roots[];
			const auto poly = roots + blockDim.x;

			complex exp_val = 1.0, value = roots[threadIdx.x];
			for (int exp = 1; exp < blockDim.x; exp <<= 1)
			{
				if (exp & threadIdx.x)
					exp_val *= value;
				value *= value;
			}

			value = 1.0;
			complex prime{};
			for (int id = threadIdx.x; ; )
			{
				prime += (id + 1) * poly[id] * exp_val;

				exp_val *= roots[threadIdx.x];
				value += poly[id] * exp_val;

				if (++id == dim)
				{
					id = 0;
					exp_val = 1.0;
				}

				if (id == threadIdx.x)
					return value / prime;
			}
		}
	}
	
	__device__ complex solveAberth(
		const int dim,
		thrust::complex<double> highest_coeff,
		const int iter_count )
	{
		extern __shared__ complex roots[];
		
		double magnitude = 1.0;
		roots[threadIdx.x] =
			thrust::polar(magnitude, (double)threadIdx.x / dim);
		__syncthreads();

		complex delta = {};
		for (int it = 0; it < iter_count; ++it)
		{
			if (threadIdx.x < dim)
			{
				delta = evalDelta(dim);

				auto root = roots[threadIdx.x];
				complex amend = {};
				for (int id = threadIdx.x + 1; ; ++id)
				{
					if (id == dim)
						id = 0;
					if (id == threadIdx.x)
						break;

					amend -= 1.0 / (roots[id] - root);
				}
				roots[threadIdx.x] -= delta / (1.0 - delta * amend);
			}
			__syncthreads();
		}
		return delta;
	}
}