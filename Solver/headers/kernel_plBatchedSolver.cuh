#pragma once

#include <thrust/complex.h>

#include <laplace/format.h>
#include <laplace/transform.cuh>

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

	struct batchedResult
	{
		batchedTerm data;
		batchedStatus status;
	};

	template <numer::laplace::transformType type, typename ... Args>
	__global__ void kernelBatchedVecSolver(
		const thrust::complex<double>* grid,
		thrust::complex<double>* glb_buf,
		batchedResult* result_grid,
		Args ... args);
}