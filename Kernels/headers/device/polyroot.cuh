#pragma once

#include <thrust/complex.h>

namespace pl
{
	__device__ double solveAberth(
		const int dim,
		const int iter_count = 100);
}