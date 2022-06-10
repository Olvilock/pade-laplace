#pragma once

#include <complex>
#include <vector>

namespace pl
{
	struct Node
	{
		double point;
		std::complex<double> value;
	};
	
	using dataset_type = std::vector<Node>;

	struct Approximation
	{
		struct Term
		{
			std::complex<double> amp;
			std::complex<double> exp;
		};
		
		std::vector<Term> terms;
	};
	
	enum class transformType {
		Trapezia, Spline
	};

	template <transformType>
	Approximation solveBatched(const dataset_type& data, unsigned depth);
}