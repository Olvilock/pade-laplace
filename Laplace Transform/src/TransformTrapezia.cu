#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <TransformTrapezia.cuh>

namespace lpl
{
	namespace
	{
		__global__ void kernelTransformTrapezia (
			const Point* points, unsigned count,
			const thrust::complex<double>* grid,
			thrust::complex<double>* res_grid );
	}

	Point::Point(const it::Point& ip) :
		point(ip.point), value(ip.value) {}

	TransformTrapezia::TransformTrapezia(const pl::dataset_type& data) :
		m_points(data) {}

	TransformTrapezia::grid_type TransformTrapezia::transformGrid
		(const grid_type& grid, unsigned depth) const
	{
		grid_type result(grid.size() * depth);

		kernelTransformTrapezia <<< grid.size(), depth >>>
			(m_points.data().get(), m_points.size(),
			 grid.data().get(),
			 result.data().get());

		return result;
	}

	namespace
	{
		__device__
		double get_exp(double base)
		{
			double exp_val = 1.0;
			for (int exp = 1; exp < blockDim.x; exp <<= 1)
			{
				if (exp & threadIdx.x)
					exp_val *= base;
				base *= base;
			}
			return exp_val;
		}

		__global__
		void kernelTransformTrapezia (
			const Point* points, unsigned count,
			const thrust::complex<double>* grid,
			thrust::complex<double>* res_grid )
		{
			thrust::complex<double>	result = {};

			if (count)
			{
				thrust::complex<double> s = grid[blockIdx.x];

				Point left = *points++;
				thrust::complex<double>
					left_exp = get_exp(-left.point) *
						   thrust::exp(-left.point * s);

				while (--count)
				{
					Point right = *points++;
					thrust::complex<double>
						right_exp = get_exp(-right.point) *
								thrust::exp(-right.point * s);

					result += 0.5 * (right.point - left.point) *
						(left_exp * left.value + right_exp * right.value);

					left = right;
					left_exp = right_exp;
				}
			}

			res_grid[threadIdx.x + blockDim.x * blockIdx.x] = result;
		}
	}
}