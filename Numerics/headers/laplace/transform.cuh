#pragma once

#include "format.h"
#include <Point.h>

#include <thrust/pair.h>
#include <thrust/complex.h>

namespace numer {
    namespace laplace
    {
        struct Point
        {
            double point;
            thrust::complex<double> value;

            __host__ __device__
                Point(const numer::Point&);
            Point() = default;
        };
		
        struct SplineSegment
        {
            double pivot;
            thrust::complex<double> d;
        };
        struct SplineEndpoint
        {
            thrust::complex<double> a, b;
        };

        template <transformType type, typename ... Args>
        __device__ thrust::pair<thrust::complex<double>, thrust::complex<double>>
            transform(
                const thrust::complex<double> s,
                Args ... args);
    }
}