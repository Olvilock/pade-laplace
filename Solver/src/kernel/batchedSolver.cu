#include <kernel/batchedSolver.cuh>

#include <device/polyroot.cuh>
#include <device/pl_slae.cuh>
#include <device/residue.cuh>

#include <device/transform.cuh>
#include <transformTypes.cuh>

namespace pl
{
	using complex = thrust::complex<double>;
	
	__device__
		const batchedStatus& operator |= (
			batchedStatus& me, const batchedStatus& other)
	{
		me = static_cast<batchedStatus>(
			static_cast<int>(me) | static_cast<int>(other));
		return me;
	}
	
	template __global__ void kernelBatchedVecSolver(
		const complex* grid,
		batchedResult* result_grid,
		TrapeziaData);
	template __global__ void kernelBatchedVecSolver(
		const complex* grid,
		batchedResult* result_grid,
		SplineData);

	template <typename Arg>
	__global__ void kernelBatchedVecSolver(
		const complex* grid,
		batchedResult* result_grid,
		Arg arg)
	{
		extern __shared__ complex roots[];
		auto coeff = roots + blockDim.x;
		auto taylor = coeff + blockDim.x;
		
		auto point = grid[blockIdx.x];
		auto transform_pair = transform(point, arg);
		taylor[threadIdx.x] = transform_pair.first;
		taylor[threadIdx.x + blockDim.x] = transform_pair.second;

		//printf("%i: %le %le\n", threadIdx.x,
		//	taylor[threadIdx.x].real(), taylor[threadIdx.x].imag());

		batchedResult* out_ptr = result_grid +
			blockIdx.x * blockDim.x * (blockDim.x + 1) / 2;
		for (int lc_dim = 1; lc_dim <= blockDim.x; out_ptr += lc_dim++)
		{
			batchedStatus status = batchedStatus::ok;
			__syncthreads();

			if (!(pl_slae_cg(lc_dim, 1000) < 1e-6))
				status |= batchedStatus::degenerate_system;

			auto this_coeff = taylor[threadIdx.x];
			if (threadIdx.x < lc_dim && threadIdx.x)
			{
				int id_sum = threadIdx.x - 1;
				int id = id_sum / 2;
					
				if (id_sum % 2)
					this_coeff +=
						taylor[id_sum - id] * coeff[id];
				this_coeff +=
					taylor[id] * coeff[id_sum - id];
					
				while (id--)
					this_coeff +=
						taylor[id] * coeff[id_sum - id] +
						taylor[id_sum - id] * coeff[id];
			}
			
			auto highest_coeff = coeff[threadIdx.x];
			
			if (!(solveAberth(lc_dim, 100) < 1e-6))
				status |= batchedStatus::Aberth_divergence;

			if (threadIdx.x == lc_dim - 1)
				coeff[0] = highest_coeff;
			__syncthreads();
			for (int id = 1; id < lc_dim; id <<= 1)
			{
				if (threadIdx.x < id && threadIdx.x + id < lc_dim)
					coeff[threadIdx.x + id] = coeff[threadIdx.x];
				__syncthreads();
			}
			highest_coeff = coeff[threadIdx.x];

			coeff[threadIdx.x] = this_coeff;
			__syncthreads();
			
			if (threadIdx.x < lc_dim)
				out_ptr[threadIdx.x] = {
				{
					numer::residue(lc_dim) / highest_coeff,
					roots[threadIdx.x] + point
				}, status };
		}
	}
}