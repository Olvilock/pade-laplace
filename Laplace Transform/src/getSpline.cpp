#include <getSpline.h>

import Spline;

namespace lpl
{
	it::Spline::storage_type getSpline(const pl::dataset_type& data)
	{
		return it::Spline(data).get_spline();
	}
}