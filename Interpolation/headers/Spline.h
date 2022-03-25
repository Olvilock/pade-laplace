#pragma once

#ifdef __CUDACC__

#include "Point.h"
#include <vector>

#else

import "Point.h";
import <vector>;

#endif

namespace it
{
	struct Cubic
	{
		std::complex<double> a, b, c, d;
	};

	struct Spline
	{
		using storage_type = std::vector<Cubic>;
		using vertices_type = std::vector<double>;

		explicit Spline(const std::vector<Point>& data);
		std::complex<double> operator () (double point) const;

		const storage_type& get_spline() const&;
		storage_type&& get_spline() &&;
	protected:
		storage_type m_spline;
		vertices_type m_vertices;
	};
}