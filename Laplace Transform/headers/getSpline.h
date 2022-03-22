#pragma once

#ifdef __CUDACC__

#include <Spline.h>
#include <plFormat.h>

#else

import <Spline.h>;
import <plFormat.h>;

#endif

namespace lpl
{
	it::Spline::storage_type getSpline(const pl::dataset_type&);
}