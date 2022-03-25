#pragma once

#include <Spline.h>
#include <plFormat.h>

#include <thrust/device_vector.h>
#include <thrust/complex.h>

namespace lpl
{
    struct Cubic
    {
        thrust::complex<double> a, b, c, d;

        __host__ __device__
            Cubic(const it::Cubic&);
    };

    struct TransformSplines
    {
        using spline_type = thrust::device_vector<Cubic>;
        using dataset_type = thrust::device_vector<double>;
        using grid_type = thrust::device_vector<thrust::complex<double> >;

        TransformSplines(spline_type, dataset_type);
        TransformSplines(const pl::dataset_type&);
        grid_type transformGrid(const grid_type& points) const;
    private:
        spline_type m_spline;
        dataset_type m_vertices;
    };
}