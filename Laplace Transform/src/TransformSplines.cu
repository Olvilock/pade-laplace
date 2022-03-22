#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cassert>

#include <TransformSplines.cuh>
#include <kernelTransformSplines.cuh>
#include <getSpline.h>

namespace lpl
{
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
}