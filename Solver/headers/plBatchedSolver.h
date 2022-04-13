#pragma once

#include <plFormat.h>

namespace pl
{
	template <numer::laplace::transformType>
	Approximation solveBatched(const dataset_type& data, unsigned depth);
}