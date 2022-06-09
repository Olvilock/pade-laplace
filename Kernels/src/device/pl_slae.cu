#include <device/pl_slae.cuh>

#include <thrust/complex.h>

namespace pl
{
	using complex = thrust::complex<double>;

	namespace
	{
		__device__ void apply(const int dim, complex vec)
		{
			extern __shared__ complex error[];
			complex* roots = error + blockDim.x;
			const complex* taylor = roots + blockDim.x;

			__syncthreads();
			roots[threadIdx.x] = {};
			__syncthreads();

			if (threadIdx.x < dim)
				for (int id = threadIdx.x + 1; ; ++id)
				{
					id -= dim * (id == dim);
					roots[id] += vec * taylor[id + threadIdx.x];
					
					if (id == threadIdx.x)
						break;
				}
			__syncthreads();
		}
		__device__ void reduce(const int dim)
		{
			extern __shared__ double data[];
			
			__syncthreads();
#pragma unroll
			for (int id = 512; id; id >>= 1)
			{
				if (threadIdx.x + id < dim && threadIdx.x < id)
					data[threadIdx.x] += data[threadIdx.x + id];
				__syncthreads();
			}
#pragma unroll
			for (int id = 1; id < blockDim.x; id <<= 1)
			{
				if (threadIdx.x < id && threadIdx.x + id < blockDim.x)
					data[threadIdx.x + id] = data[threadIdx.x];
				__syncthreads();
			}
		}
	}
	
	__device__ double pl_slae_cg(const int dim, int iter_count)
	{
		extern __shared__ complex buffer[];
		complex* result = buffer + blockDim.x;
		const complex* minus_rhs = result + blockDim.x + dim;

		static_assert(sizeof(complex) >= sizeof(double),
			"sizeof(complex) must not be less than sizeof(double)");
		extern __shared__ double norm[];
		
		apply(dim, thrust::conj(-minus_rhs[threadIdx.x]));
		
		norm[threadIdx.x] = thrust::norm(result[threadIdx.x]);
		reduce(dim);

		complex error = thrust::conj(result[threadIdx.x]),
				basis = error, root{};

		auto err_norm = norm[threadIdx.x];
		
		while (iter_count--)
		{
			if (err_norm < 1e-32)
				break;

			apply(dim, basis);
			
			if (threadIdx.x < dim)
				norm[threadIdx.x] = thrust::norm(result[threadIdx.x]);
			
			reduce(dim);
			apply(dim, thrust::conj(result[threadIdx.x]));

			auto alpha = err_norm / norm[threadIdx.x];

			root += alpha * basis;
			error -= alpha * thrust::conj(result[threadIdx.x]);

			if (threadIdx.x < dim)
				norm[threadIdx.x] = thrust::norm(error);
			reduce(dim);

			auto beta = norm[threadIdx.x] / err_norm;
			basis = error + beta * basis;
			
			err_norm = norm[threadIdx.x];
		}
		
		apply(dim, root);

		if (threadIdx.x < dim)
			norm[threadIdx.x] =
				thrust::norm(result[threadIdx.x] + minus_rhs[threadIdx.x]);
		reduce(dim);
		
		if (!threadIdx.x)
			printf("Error norm^2 is %le, "
				   "computed error norm^2 %le\n", norm[threadIdx.x], err_norm);

		if (threadIdx.x < dim)
			result[dim - threadIdx.x - 1] = root;
		__syncthreads();

		return sqrt(err_norm);
	}
}