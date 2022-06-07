#pragma once

#ifdef __CUDACC__

#include <Spline.h>
#include <plFormat.h>

#else

import <Spline.h>;
import <plFormat.h>;

#endif

namespace pl
{
	numer::Spline::storage_type getSpline(const dataset_type&);
}