#pragma once

#include "plSolver.h"
#include <vector>

namespace pl
{
	struct Cubic
	{
		std::complex<double> a, b, c, d;
	};
	
	using spline_type = std::vector<Cubic>;
	spline_type getSpline(const dataset_type&);
}