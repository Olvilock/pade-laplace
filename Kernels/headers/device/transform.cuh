#pragma once

#include <thrust/pair.h>
#include <thrust/complex.h>

namespace pl
{
    template <typename Arg>
    __device__ thrust::pair<thrust::complex<double>, thrust::complex<double>>
        transform(const thrust::complex<double> s, Arg arg);
}