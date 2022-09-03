#include <device/residue.cuh>

namespace numer
{
	using complex = thrust::complex<double>;
	
	__device__ complex residue(const int dim)
	{
		extern __shared__ complex roots[];
		complex *coeff = roots + blockDim.x,
			root = roots[threadIdx.x],
			denom = 1.0;
		for (int id = threadIdx.x + 1; ; ++id)
		{
			if (id == dim)
				id = 0;
			if (id == threadIdx.x)
				break;
			denom *= root - roots[id];
		}
		
		complex
			exp_val = 1.0,
			numer = {};
		for (int exp = 1; exp <= dim; exp <<= 1)
		{
			if (exp & threadIdx.x)
				exp_val *= root;
			root *= root;
		}

		for (int id = threadIdx.x; ; )
		{
			numer += exp_val * coeff[id];
			exp_val *= roots[threadIdx.x];

			if (++id == dim)
			{
				id = 0;
				exp_val = 1.0;
			}
			if (id == threadIdx.x)
				return numer / denom;
		}
	}
}