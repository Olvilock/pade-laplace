#include <plApprox.h>
#include <plFormat.h>

#include <TransformSplines.cuh>
#include <TransformTrapezia.cuh>

#include <Point.h>

#include <thrust/complex.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <string>

namespace pl
{
	namespace
	{
		double getRealBound(const dataset_type& data)
		{
			if (data.empty())
				return 0.0;

			auto less_value = [](const it::Point& a, const it::Point& b)
						{ return std::abs(a.value) < std::abs(b.value); };

			auto it = std::max_element(data.begin(), data.end(), less_value);
			
		}
	}

	Approximation approx(dataset_type data, unsigned depth)
	{
		auto less_point = [](const it::Point& a, const it::Point& b)
							{ return a.point < b.point; };
		thrust::sort(thrust::host, data.begin(), data.end(), less_point);

		std::vector<thrust::complex<double> > points
		{
			{ -0.7, 1.0 },
			{ -0.5, 0.1 },
			{ 0.2, 1.0 },
			{ 1.0, 0.1 },
			{ 2.0, 1.0 },
			{ 3.0, 0.1 },
			{ 4.0, 1.0 },
			{ 5.0, 0.1 },
			{ 10.0, 1.0 },
			{ 100.0, 0.1 }
		};

		std::cout << "Initializing Trapezia...\n";
		lpl::TransformTrapezia laplace_trapezia(data);

		std::cout << "Transforming Trapezia...\n";
		thrust::host_vector<thrust::complex<double>>
			result_trapezia = laplace_trapezia.transformGrid(points, depth);

		std::cout << "Initializing Splines...\n";
		lpl::TransformSplines laplace_splines(data);

		std::cout << "Transforming Splines...\n";
		thrust::host_vector<thrust::complex<double>>
			result_splines = laplace_splines.transformGrid(points, depth);

		std::cout << "\nTransformed data:\n";
		for (std::size_t i = 0; i != points.size(); ++i)
		{
			std::cout << "Point: " << points[i] << "\n\n";
			auto true_res = 1.0 / (points[i] + 1);

			std::cout << "Depth | ";
			std::cout << std::setw(20) << "True result" << " | ";
			std::cout << std::setw(20) << "Trapezia result" << " | ";
			std::cout << "Splines result\n";
			for (int d = 0; d != depth; ++d)
			{
				std::cout << std::setw(5) << d << " | ";

				std::cout << true_res << " | ";
				true_res *= -(d + 1) / (points[i] + 1);

				std::cout << result_trapezia[i * depth + d] << " | ";
				std::cout << result_splines[i * depth + d] << "\n";
			}
			std::cout << "\n\n";
		}

		return { { { -1.0, 0.0 } } };
	}
}