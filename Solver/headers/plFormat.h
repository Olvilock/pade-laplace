#pragma once

#ifdef __CUDACC__

#include <Point.h>
#include <complex>
#include <vector>

#else

import <Point.h>;
import <complex>;
import <vector>;

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