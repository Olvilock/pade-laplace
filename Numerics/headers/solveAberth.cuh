#pragma once

#include <thrust/complex.h>
#include <thrust/device_vector.h>

namespace solve
{
	void Aberth(const thrust::device_vector<thrust::complex<double> >& polies, unsigned degree,
					  thrust::device_vector<thrust::complex<double> >& roots,
				double absolute = 1e-9, double epsilon = 1e-6);
}