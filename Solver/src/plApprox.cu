#include <plApprox.h>
#include <plFormat.h>
#include <TransformSplines.cuh>
#include <Point.h>

#include <thrust/complex.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include <vector>
#include <algorithm>

namespace pl
{
	double getRealBound(const dataset_type& data)
	{
		auto less_value = [](const it::Point& a, const it::Point& b)
					{ return a.value < b.value; };

		auto it = std::max_element(data.begin(), data.end(), less_value);
		return 0.0;
	}

	Approximation approx(dataset_type data)
	{
		auto less_point = [](const it::Point& a, const it::Point& b)
							{ return a.point < b.point; };
		thrust::sort(thrust::host, data.begin(), data.end(), less_point);

		lpl::TransformSplines laplace(data);
		std::vector<thrust::complex<double> > points
		{
			{ -2.0, 1.0 },
			{ -1.0, 1.0 },
			{ -0.9, 1.0 },
			{ -0.7, 1.0 },
			{ -0.5, 1.0 },
			{ 1.0, 1.0 },
			{ 1.0, -1.0 },
			{ 2.0, 1.0 },
			{ 3.0, 1.0 },
			{ 4.0, 1.0 },
			{ 5.0, 1.0 }
		};

		std::cout << "Transforming...\n";
		thrust::host_vector<thrust::complex<double>> result = laplace.transformGrid(points);

		std::cout << "\nTransformed data:\n";
		for (std::size_t i = 0; i != points.size(); ++i)
		{
			std::cout << points[i].real() << ' ' << points[i].imag() << '\n';

			auto true_res = 1.0 / (points[i] + 1.0);
			std::cout << true_res.real() << ' ' << true_res.imag() << '\n';

			std::cout << result[i].real() << ' ' << result[i].imag() << "\n\n";

		}
		return { { { -1.0, 0.0 } } };
	}
}