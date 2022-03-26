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
                delta * ( front_seg.c +
                delta * front_seg.d ) ),

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

		kernelTransformSplines <<< grid.size(), depth, depth * sizeof(double) >>>
			(m_spline.data().get(), m_spline.size(), m_leftEnd, m_rightEnd,
			 grid.data().get(), result.data().get());

		return result;
	}

	namespace
	{
        __device__ void set_pow(double base)
        {
            extern __shared__ double pow[];
            pow[threadIdx.x] = 1.0;

            for (int exp = 1; exp < blockDim.x; exp <<= 1)
            {
                if (exp & threadIdx.x)
                    pow[threadIdx.x] *= base;
                base *= base;
            }
        }

        template <int offset>
        __device__ thrust::complex<double> get_coeff(
            thrust::complex<double> s)
        {
            extern __shared__ double pow[];

            auto coeff = 1 / s;
            int next = 1;
#pragma unroll
            while (next <= offset)
                coeff *= next++ / s;

            auto result = coeff * pow[threadIdx.x];
            for (int exp = threadIdx.x; exp; ++next)
            {
                coeff *= -exp-- * next / ((next - offset) * s);
                result += coeff * pow[exp];
            }
            return result;
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

            set_pow(-segment.pivot);
            auto exp = thrust::exp(-segment.pivot * s);
            auto result = exp *
                (   get_coeff<0>(s) * left.a +
                    get_coeff<1>(s) * left.b );
            
            while (--segments_count)
            {
                auto old_cubic = segment.d;
                segment = *segments++;
                result += exp *
                    get_coeff<3>(s) * (segment.d - old_cubic);

                set_pow(-segment.pivot);
                exp = thrust::exp(-segment.pivot * s);
            }

            result -= exp *
                (   get_coeff<0>(s) * right.a +
                    get_coeff<1>(s) * right.b +
                    get_coeff<3>(s) * segment.d );

            res_grid[glb_id] = result;
        }
	}
}