#pragma once

#include <thrust/complex.h>

namespace numer
{
	__device__ thrust::complex<double> solveAberth(
		const int dim,
		thrust::complex<double> highest_coeff,
		const int iter_count = 100);
}