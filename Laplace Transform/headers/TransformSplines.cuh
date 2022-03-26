#pragma once

#include <plFormat.h>
#include <Spline.h>

#include <thrust/device_vector.h>
#include <thrust/complex.h>

namespace lpl
{
    struct SplineSegment
    {
        double right;
        thrust::complex<double> cubic;
    };

    struct SplineEndpoint
    {
        thrust::complex<double> value;
        thrust::complex<double> slope;
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