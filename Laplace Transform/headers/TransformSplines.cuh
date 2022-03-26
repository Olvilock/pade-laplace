#pragma once

#include <plFormat.h>
#include <Spline.h>

#include <thrust/device_vector.h>
#include <thrust/complex.h>

namespace lpl
{
    struct SplineSegment
    {
        double pivot;
        thrust::complex<double> d;
    };

    struct SplineEndpoint
    {
        thrust::complex<double> a, b;
    };

    struct TransformSplines
    {
        using dataset_type = thrust::device_vector<SplineSegment>;
        using grid_type = thrust::device_vector<thrust::complex<double> >;

        TransformSplines(const pl::dataset_type&);
        grid_type transformGrid(const grid_type& points, unsigned depth) const;
    private:
        dataset_type m_spline;
        SplineEndpoint m_leftEnd, m_rightEnd;
    };
}