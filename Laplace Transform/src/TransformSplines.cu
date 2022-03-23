#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <TransformSplines.cuh>
#include <Spline.h>
#include <getSpline.h>

#include <cassert>

namespace lpl
{
    namespace
    {
        __global__
        void kernelTransformSplines(const it::Cubic* polies, size_t polies_count,
                const double* vertices,
                const thrust::complex<double>* grid, size_t grid_size,
                thrust::complex<double>* res_grid);
    }

	TransformSplines::TransformSplines(spline_type spline, dataset_type vertices) :
		m_spline(std::move(spline)), m_vertices(std::move(vertices))
	{
		assert(m_spline.size() + 1 == m_vertices.size());
	}
	TransformSplines::TransformSplines(const pl::dataset_type& data) :
		m_spline(getSpline(data)), m_vertices(data.size())
	{
		for (std::size_t id = 0; id != data.size(); ++id)
			m_vertices[id] = data[id].point;
		assert(m_spline.size() + 1 == m_vertices.size());
	}

	TransformSplines::grid_type TransformSplines::transformGrid(const grid_type& grid) const
	{
		grid_type result(m_vertices.size());

		constexpr int blk_size = 64;
		kernelTransformSplines <<<(m_vertices.size() - 1) / blk_size + 1, blk_size>>>
			(m_spline.data().get(), m_spline.size(), m_vertices.data().get(),
			 grid.data().get(), grid.size(), result.data().get());

		return result;
	}

	namespace
	{
        __global__
        void kernelTransformSplines(const it::Cubic* polies, size_t polies_count,
                const double* vertices,
                const thrust::complex<double>* grid, size_t grid_size,
                thrust::complex<double>* res_grid)
        {
            size_t id = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
            if (id >= grid_size)
                return;
            if (!polies_count)
            {
                res_grid[id] = {};
                return;
            }

            double vertex = *vertices++;
            thrust::complex<double>
                s = grid[id],
                coeff = thrust::exp(-s * vertex) / s,
                result = {};
            while (polies_count--)
            {
                it::Cubic poly = *polies++;
                double delta = vertex - (vertex = *vertices++);

                result += coeff * (
                    poly.a + delta * (poly.b + delta * (poly.c + delta * poly.d))
                    + (poly.b + delta * (poly.c * 2 + delta * poly.d * 3)) / s
                    + (poly.c * 2 + (delta + 1 / s) * poly.d * 6) / (s * s));

                coeff = thrust::exp(-s * vertex) / s;

                result -= coeff * (poly.a
                    + (poly.b + (poly.c * 2 +
                        poly.d * 6 / s) / s) / s);
            }
            res_grid[id] = result;
        }
	}
}