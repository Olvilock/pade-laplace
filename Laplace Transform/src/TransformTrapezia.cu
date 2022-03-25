#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <TransformTrapezia.cuh>

namespace lpl
{
	namespace
	{
		__global__
		void kernelTransformTrapezia(const Point* points, unsigned count,
									 const thrust::complex<double>* grid,
									 thrust::complex<double>* res_grid);
	}

	Point::Point(const it::Point& ip) :
		point(ip.point), value(ip.value) {}

	TransformTrapezia::TransformTrapezia(const pl::dataset_type& data) :
		m_points(data) {}

	TransformTrapezia::grid_type TransformTrapezia::transformGrid
		(const grid_type& grid, unsigned depth) const
	{
		grid_type result(grid.size() * depth);

		constexpr int blk_size = 64;
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
		void kernelTransformTrapezia(const Point* points, unsigned count,
									 const thrust::complex<double>* grid,
									 thrust::complex<double>* res_grid)
		{
			thrust::complex<double>
				s = grid[blockIdx.x],
				result = {};

			if (count)
			{
				Point left = *points++;
				double left_exp = get_exp(-left.point);
				while (--count)
				{
					Point right = *points++;
					double right_exp = get_exp(-right.point);

					result += 0.5 * (right.point - left.point) *
						( left_exp * left.value * thrust::exp(-s * left.point)
						+ right_exp * right.value * thrust::exp(-s * right.point));

					left = right;
					left_exp = right_exp;
				}
			}

			res_grid[threadIdx.x + blockDim.x * blockIdx.x] = result;
		}
	}
}