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
		thrust::tie(
			taylor[threadIdx.x],
			taylor[threadIdx.x + blockDim.x]) = transform(point, arg);

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

			if (!(solveAberth(lc_dim, 100) < 1e-6))
				status |= batchedStatus::Aberth_divergence;

			coeff[threadIdx.x] = this_coeff;
			if (threadIdx.x < lc_dim)
				out_ptr[threadIdx.x] = {
				{
					numer::residue(lc_dim) / coeff[lc_dim - 1],
					roots[threadIdx.x] + point
				}, status };
		}
	}
}