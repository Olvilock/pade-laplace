#pragma once

#ifdef __CUDACC__

#include <Point.h>
#include <laplace/format.h>

#include <complex>
#include <vector>

#else

import <Point.h>;
import <laplace/format.h>;

import <complex>;
import <vector>;

#endif

namespace pl
{
	using numer::laplace::transformType;
	using dataset_type = std::vector<numer::Point>;

	struct Term
	{
		std::complex<double> amp;
		std::complex<double> exp;
	};

	struct Approximation
	{
		std::vector<Term> points;
	};
}