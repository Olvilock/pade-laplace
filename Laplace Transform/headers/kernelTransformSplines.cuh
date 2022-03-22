#pragma once

#include <device_launch_parameters.h>

#include <Spline.h>
#include <thrust/complex.h>

namespace lpl
{
    __global__
    void kernelTransformSplines(const it::Cubic* polies, size_t polies_count,
                                const double* vertices,
                                const thrust::complex<double>* grid, size_t grid_size,
                                thrust::complex<double>* res_grid);
}