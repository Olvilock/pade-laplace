#pragma once

#include <thrust/complex.h>

namespace numer
{
	__device__ thrust::complex<double> residue(const int dim);
}