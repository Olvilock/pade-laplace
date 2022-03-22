#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <kernelTransformSplines.cuh>

namespace lpl
{
    __global__
    void kernelTransformSplines(const it::Cubic* polies, size_t polies_count,
                                const double* vertices,
                                const thrust::complex<double>* grid, size_t grid_size,
                                thrust::complex<double>* res_grid)
    {
        size_t id = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
        if (id >= grid_size)
            return;
        if (!polies_count)
        {
            res_grid[id] = {};
            return;
        }

        double vertex = *vertices++;
        thrust::complex<double>
            s = grid[id],
            coeff = thrust::exp(-s * vertex) / s,
            result = {};
        while (polies_count--)
        {
            it::Cubic poly = *polies++;
            double delta = vertex - (vertex = *vertices++);

            result += coeff * (
                   poly.a + delta * (poly.b + delta * (poly.c + delta * poly.d))
                + (poly.b + delta * (poly.c * 2 + delta * poly.d * 3)) / s
                + (poly.c * 2 + (delta + 1 / s) * poly.d * 6) / (s * s) );
            
            coeff = thrust::exp(-s * vertex) / s;
            
            result -= coeff * (poly.a
                  + (poly.b + (poly.c * 2 +
                     poly.d * 6 / s) / s) / s);
        }
        res_grid[id] = result;
    }
}