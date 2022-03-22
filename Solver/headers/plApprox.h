#pragma once

#include <Point.h>
#include "plFormat.h"

#include <vector>

namespace pl
{
	struct Approximation
	{
		std::vector<Term> points;
	};

	Approximation approx(dataset_type data);
}