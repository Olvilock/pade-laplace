#pragma once

#include <thrust/complex.h>

namespace pl
{
	struct batchedTerm
	{
		thrust::complex<double> amp;
		thrust::complex<double> exp;
	};
	
	enum class batchedStatus {
		ok = 0b000,
		degenerate_system = 0b001,
		Aberth_divergence = 0b010,
		untouched = 0b100
	};
	__device__ const batchedStatus& operator |= (
		batchedStatus& me, const batchedStatus& other);

	struct batchedResult
	{
		batchedTerm data;
		batchedStatus status;
	};

	template <typename Arg>
	__global__ void kernelBatchedVecSolver(
		const thrust::complex<double>* grid,
		batchedResult* result_grid,
		Arg arg);
}