#pragma once

#include <thrust/complex.h>

namespace pl
{
	__device__ double pl_slae_cg(const int dim, int iter_count);
}