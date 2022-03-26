#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <TransformSplines.cuh>
#include <getSpline.h>

#include <iostream>
#include <cassert>

namespace lpl
{
    namespace
    {
        __global__ void kernelTransformSplines (
            const SplineSegment* segments, unsigned segments_count,
            const SplineEndpoint left,
            const SplineEndpoint right,

            const thrust::complex<double>* grid,
            thrust::complex<double>* res_grid );
    }

	TransformSplines::TransformSplines(const pl::dataset_type& data)
    {
        if (data.size() < 2)
            return;
        m_spline.reserve(data.size());
        auto spline = getSpline(data);
        {
            auto front_seg = spline.front();
            auto delta = data[0].point - data[1].point;
            m_leftEnd = { front_seg.a +
                delta * ( front_seg.b +
                delta * ( front_seg.c + delta * front_seg.d)),

                          front_seg.b +
                delta * (2.0 * front_seg.c +
                delta *  3.0 * front_seg.d) };
        }
        m_spline.push_back({ data.front().point });
        for (std::size_t id = 1; id != data.size(); ++id)
            m_spline.push_back({ data[id].point, spline[id - 1].d });
        
        {
            auto back_seg = spline.back();
            m_rightEnd = { back_seg.a, back_seg.b };
        }
    }

	TransformSplines::grid_type TransformSplines::transformGrid
        (const grid_type& grid, unsigned depth) const
	{
		grid_type result(grid.size() * depth);

		kernelTransformSplines <<< grid.size(), depth,
                                  (depth + splineDim) * sizeof(double) >>>
			(m_spline.data().get(), m_spline.size(), m_leftEnd, m_rightEnd,
			 grid.data().get(), result.data().get());

		return result;
	}

	namespace
	{
        __device__ void get_exp(double base, unsigned id)
        {
            extern __shared__ double exp_val[];
            exp_val[id] = 1.0;

            for (int cur_exp = 1; cur_exp <= id; cur_exp <<= 1)
            {
                if (cur_exp & id)
                    exp_val[id] *= base;
                base *= base;
            }
        }
        __device__ thrust::complex<double> get_coeff(
            thrust::complex<double> s, double t, unsigned offset)
        {
            extern __shared__ double exp[];

            for (unsigned id = threadIdx.x;
                    id < blockDim.x + offset;
                    id += blockDim.x)
                get_exp(t, id);

            offset += threadIdx.x;
            auto coeff = thrust::exp(-t * s) / s,
                result = exp[offset] * coeff;

            while (offset)
            {
                coeff *= offset-- / s;
                result += coeff * exp[offset];
            }

            return (1 - 2 * (threadIdx.x % 2)) * result;
        }

        __global__ void kernelTransformSplines (
            const SplineSegment* segments, unsigned segments_count,
            const SplineEndpoint left,
            const SplineEndpoint right,

            const thrust::complex<double>* grid,
            thrust::complex<double>* res_grid )
        {
            const auto glb_id = threadIdx.x + blockDim.x * blockIdx.x;
            const auto grid_id = blockIdx.x;

            if (!segments_count)
            {
                res_grid[glb_id] = {};
                return;
            }

            auto s = grid[grid_id];
            auto segment = *segments++;
            auto result =
                get_coeff(s, segment.right, 0) * left.value +
                get_coeff(s, segment.right, 1) * left.slope;
            
            while (--segments_count)
            {
                auto old_segment = segment;
                segment = *segments++;
                result +=
                    get_coeff(s, old_segment.right, splineDim) *
                        (segment.cubic - old_segment.cubic);
            }

            result -=
                get_coeff(s, segment.right, 0) * right.value +
                get_coeff(s, segment.right, 1) * right.slope +
                get_coeff(s, segment.right, splineDim) * segment.cubic;

            res_grid[glb_id] = result;
        }
	}
}