#pragma once

#include <plFormat.h>
#include <Point.h>

#include <thrust/device_vector.h>
#include <thrust/complex.h>

namespace lpl
{
	struct Point
	{
		double point;
		thrust::complex<double> value;

		__host__ __device__
		Point(const it::Point&);
	};

	struct TransformTrapezia
	{
		using dataset_type = thrust::device_vector<Point>;
		using grid_type = thrust::device_vector<thrust::complex<double> >;

		TransformTrapezia(const pl::dataset_type&);
		grid_type transformGrid (const grid_type&, unsigned depth) const;
	private:
		dataset_type m_points;
	};
}