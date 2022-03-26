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

		kernelTransformSplines <<< grid.size(), depth, depth * sizeof(double) >>>
			(m_spline.data().get(), m_spline.size(), m_leftEnd, m_rightEnd,
			 grid.data().get(), result.data().get());

		return result;
	}

	namespace
	{
        __device__ void get_exp(double base)
        {
            extern __shared__ double exp_val[];
            exp_val[threadIdx.x] = 1.0;

            for (int exp = 1; exp < blockDim.x; exp <<= 1)
            {
                if (exp & threadIdx.x)
                    exp_val[threadIdx.x] *= base;
                base *= base;
            }
        }
        __device__ thrust::complex<double> get_coeff(
            thrust::complex<double> s, double mt)
        {
            extern __shared__ double exp[];
            get_exp(mt);

            thrust::complex<double>
                coeff = { 1.0 },
                result = exp[threadIdx.x];
            for (int i = threadIdx.x; i; )
            {
                coeff *= -i-- / s;
                result += coeff * exp[i];
            }

            return result * thrust::exp(mt * s);
        }

        __global__ void kernelTransformSplines (
            const SplineSegment* segments, unsigned segments_count,
            const SplineEndpoint left,
            const SplineEndpoint right,

            const thrust::complex<double>* grid,
            thrust::complex<double>* res_grid )
        {
            const auto glb_id = threadIdx.x + blockDim.x * blockIdx.x;
            const auto lc_id = threadIdx.x;

            if (!segments_count)
            {
                res_grid[glb_id] = {};
                return;
            }

            auto segment = *segments++;
            thrust::complex<double>
                s = grid[blockIdx.x],
                coeff_s_powm4 = 6//(lc_id + 1) * (lc_id + 2) * (lc_id + 3)
                              / ((s * s) * (s * s)),
                result = get_coeff(s, -segment.right) *
                    (left.value + (lc_id + 1) * left.slope / s) / s;

            while (--segments_count)
            {
                auto old_segment = segment;
                segment = *segments++;
                result += get_coeff(s, -old_segment.right) *
                    (segment.cubic - old_segment.cubic) * coeff_s_powm4;
            }
            
            result -= get_coeff(s, -segment.right) *
                ( (right.value + (lc_id + 1) * right.slope / s) / s
                    - segment.cubic * coeff_s_powm4);

            res_grid[glb_id] = result;
        }
	}
}