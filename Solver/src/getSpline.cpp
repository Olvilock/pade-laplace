#include <getSpline.h>

import Spline;

namespace pl
{
	numer::Spline::storage_type getSpline(const pl::dataset_type& data)
	{
		return numer::Spline(data).get_spline();
	}
}