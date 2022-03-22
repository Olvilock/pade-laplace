#pragma once

#ifdef __CUDACC__

#include <Point.h>
#include <vector>
#include <complex>

#else

import <complex>;
import <vector>;
import <Point.h>;

#endif

namespace pl
{
	using dataset_type = std::vector<it::Point>;

	struct Term
	{
		double amp;
		std::complex<double> exp;
	};
}