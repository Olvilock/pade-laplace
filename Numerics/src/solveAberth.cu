#include <solveAberth.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cassert>

namespace solve
{
	namespace
	{
		using complex = thrust::complex<double>;

		__global__ void kernelAberth(const complex* polies, complex* solutions,
									 double abs, double eps);
	}

	void Aberth(const thrust::device_vector<complex>& polies, unsigned degree,
					  thrust::device_vector<complex>& roots,
				double absolute, double epsilon)
	{
		assert(polies.size() % (degree + 1) == 0);
		assert(roots.size() % degree == 0);
		assert(polies.size() / (degree + 1) == roots.size() / degree);

		kernelAberth <<< roots.size() / degree, degree,
						(degree * 2 + 1) * sizeof(complex) >>>
			(polies.data().get(), roots.data().get(), absolute, epsilon);
	}

	namespace
	{
		struct Eval
		{
			complex value;
			complex prime;
		};

		__device__
		Eval evaluate(complex point)
		{
			extern __shared__ complex poly[];

			complex exp_val = { 1.0 }, value = point;
			for (int exp = 1; exp < blockDim.x; exp <<= 1)
			{
				if (exp & threadIdx.x)
					exp_val *= value;
				value *= value;
			}

			value = {};
			complex prime{};
			for (int id = threadIdx.x + 1; ; ++id)
			{
				prime += id * poly[id] * exp_val;

				exp_val *= point;
				value += poly[id] * exp_val;

				if (id == blockDim.x)
				{
					id = 0;
					exp_val = { 1.0 };
					value += poly[0];
				}
				if (id == threadIdx.x)
					break;
			}

			return { value, prime };
		}

		__global__ void kernelAberth(const complex* polies, complex* solutions,
									 double abs, double eps)
		{
			int local_id = threadIdx.x;
			int global_id = threadIdx.x + (blockDim.x + 1) * blockIdx.x;

			polies += global_id;
			solutions += global_id;

			extern __shared__ complex poly[];
			poly[local_id] = *polies;
			if (!local_id)
				poly[blockDim.x] = polies[blockDim.x];

			complex* sols = poly + blockDim.x + 1;
			complex sol = *solutions;
			__syncthreads();

			for (int it = 0; it != 30; ++it)
			{
				Eval eval = evaluate(sol);
				if (thrust::abs(eval.value) < min(abs, eps * thrust::abs(eval.prime)))
					break;

				sols[local_id] = sol;
				__syncthreads();

				complex delta = eval.prime / eval.value;
				for (int id = local_id + 1; ; ++id)
				{
					if (id == blockDim.x)
						id = 0;
					if (id == local_id)
						break;

					delta += 1.0 / (sols[id] - sol);
				}
				sol -= 1.0 / delta;
			}
			*solutions = sol;
		}
	}
}