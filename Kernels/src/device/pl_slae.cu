#include <device/pl_slae.cuh>

#include <thrust/complex.h>

namespace pl
{
	using complex = thrust::complex<double>;
	
	__device__ double slae_LU(const int dim)
	{
		/*
		extern __shared__ complex buffer[];
		if (threadIdx.x < dim)
			for (int col = 0; col <= dim; ++col)
				glb_buf[dim * col + threadIdx.x] =
					buffer[col + threadIdx.x];
		__syncthreads();
		
		for (int row = 0; row < dim; ++row)
		{
			if (threadIdx.x < dim && threadIdx.x != row)
			{
				auto col_ptr = glb_buf + dim * row;
				auto coeff = col_ptr[threadIdx.x] / col_ptr[row];
				
				col_ptr = glb_buf;
				for (int col = 0; col <= dim; ++col)
				{
					col_ptr[threadIdx.x] -= coeff * col_ptr[row];
					col_ptr += dim;
				}
			}
			__syncthreads();
		}
		
		double error = {};
		if (threadIdx.x < dim)
		{
			auto diff = buffer[dim + threadIdx.x];
			glb_buf[dim * dim + threadIdx.x] /=
				-glb_buf[(dim + 1) * threadIdx.x];
			for (int id = 0; id < dim; ++id)
				diff += glb_buf[dim * dim + id] *
					buffer[id + threadIdx.x];
			
			double norm = {};
			for (int id = 0; id < dim; ++id)
				norm += thrust::abs(glb_buf[dim * id + threadIdx.x]);
			error = thrust::abs(diff * norm);
		}
		__syncthreads();

		if (threadIdx.x < dim)
			buffer[blockDim.x + dim - threadIdx.x - 1] =
				glb_buf[dim * dim + threadIdx.x];
		__syncthreads();
		
		return error;*/
		
		return {};
	}
}