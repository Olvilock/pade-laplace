#include <kernel_plBatchedSolver.cuh>

#include <polyroot.cuh>
#include <pl_slae.cuh>
#include <residue.cuh>

namespace pl
{
	using complex = thrust::complex<double>;
	using numer::laplace::transformType;
	using numer::laplace::Point;
	
	__device__
		const batchedStatus& operator |= (
			batchedStatus& me, const batchedStatus& other)
	{
		me = static_cast<batchedStatus>(
			static_cast<int>(me) | static_cast<int>(other));
		return me;
	}

	template
	__global__ void kernelBatchedVecSolver<transformType::Trapezia>(
		const complex*, complex*, batchedResult*,
		const Point*, unsigned);
	template
	__global__ void kernelBatchedVecSolver<transformType::Spline>(
		const complex*, complex*, batchedResult*,
		const numer::laplace::SplineSegment*, unsigned,
		const numer::laplace::SplineEndpoint,
		const numer::laplace::SplineEndpoint);
	
	template <transformType type, typename ... Args>
	__global__ void kernelBatchedVecSolver(
		const complex* grid,
		complex* glb_buf,
		batchedResult* result_grid,
		Args ... args)
	{
		auto s = grid[blockIdx.x];
		auto taylor = numer::laplace::transform<type>(s, args...);

		glb_buf += blockIdx.x * blockDim.x * (blockDim.x + 1);

		batchedResult* out_ptr = result_grid +
			blockIdx.x * blockDim.x * (blockDim.x + 1) / 2;
		__syncthreads();
		for (int lc_dim = 1; lc_dim <= blockDim.x; out_ptr += lc_dim++)
		{
			extern __shared__ complex buffer[];
			buffer[threadIdx.x] = taylor.first;
			buffer[threadIdx.x + blockDim.x] = taylor.second;
			__syncthreads();

			batchedStatus status = batchedStatus::ok;
			if (!(slae_LU(lc_dim, glb_buf) < 1e-6))
				status |= batchedStatus::degenerate_system;
//			if (threadIdx.x < lc_dim)
//				printf("dim = %i, id = %i, diff = %lf\n",
//					lc_dim, threadIdx.x, slae_res);

			auto num_coeff = taylor.first;
			complex highest_coeff;
			if (threadIdx.x < lc_dim)
			{
				if (threadIdx.x)
				{
					int id_sum = threadIdx.x - 1;
					int id = id_sum / 2;
					
					num_coeff +=
						buffer[id] * buffer[blockDim.x + id_sum - id];
					if (id * 2 != id_sum)
						num_coeff +=
							buffer[id_sum - id] * buffer[blockDim.x + id];
					while (id--)
						num_coeff +=
							buffer[id] * buffer[blockDim.x + id_sum - id] +
							buffer[id_sum - id] * buffer[blockDim.x + id];
				}
				highest_coeff = buffer[blockDim.x + lc_dim - 1];
			}

			if (!(thrust::abs(numer::solveAberth(lc_dim, highest_coeff)) < 1e-6))
				status |= batchedStatus::Aberth_divergence;

			buffer[threadIdx.x + blockDim.x] = num_coeff;
			if (threadIdx.x < lc_dim)
				out_ptr[threadIdx.x] = {
				{
					numer::residue(lc_dim) / highest_coeff,
					buffer[threadIdx.x] + s
				}, status };
		}
	}
}